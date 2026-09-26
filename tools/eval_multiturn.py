"""多轮/连续对话能力检测（评测任务，只测不改）。

四套测试：
- A 跨轮记忆与指代：首轮注入信息（姓名/数字/偏好），后续轮追问召回，量化召回正确率；
- B 上下文长度压力：构造 256~2048 token 的合成多轮历史（训练边界 768 内外），量化重复率/跑题率；
- C 长程多轮稳定性：单 session 连续 12 轮，逐轮记录长度漂移、重复率、角色漂移、延迟；
- D 长度边界行为：超 32768 token prompt 的越界实际表现（守卫缺口验证）。

模型：ours dpo_768（主测）、ours full_sft_768、official full_sft_768（对照）。
产出：out/multiturn_eval.md + out/multiturn_samples/*.md

用法：uv run python tools/eval_multiturn.py --device cuda:0
"""

from __future__ import annotations

import argparse
import re
import statistics
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
OFFICIAL_DIR = REPO / "out" / "official"
SAMPLES_DIR = REPO / "out" / "multiturn_samples"

SEED = 42


def ngram_rep_rate(text: str, n: int = 3) -> float:
    cleaned = re.sub(r"\s", "", text)
    if len(cleaned) < n + 1:
        return 0.0
    grams = [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


class MultiTurnSession:
    """与 ChatSession 相同的多轮语义（渲染→截断→生成→eos 尾截→回填历史），并记录逐轮指标。"""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        official: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 192,
    ):
        self.official = official
        self.model = model
        self.tokenizer = tokenizer
        self.history: list[dict[str, str]] = []
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.turns: list[dict] = []

    def respond(self, user_text: str) -> str:
        """多轮语义与 ChatSession 一致；生成内核按实现分流——
        我们的模型走 inference/generator（预分配 cache），官方模型走其自带 generate()（对照同口径）。"""
        self.history.append({"role": "user", "content": user_text})
        prompt = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        torch.manual_seed(SEED)
        t0 = time.perf_counter()
        if self.official:
            out = self.model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            gen_ids = out[0][inputs["input_ids"].shape[1] :]
        else:
            from minimind_reborn.configuration.schemas import GenerateConfig
            from minimind_reborn.inference.generator import generate

            out = generate(
                self.model,
                inputs["input_ids"],
                GenerateConfig(temperature=self.temperature, top_p=self.top_p, max_new_tokens=self.max_new_tokens),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                attention_mask=inputs["attention_mask"],
            )
            gen_ids = out["generated_ids"][0]
        elapsed = time.perf_counter() - t0
        text = self.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
        eos = self.tokenizer.eos_token
        if eos and eos in text:
            text = text.split(eos)[0]
        self.history.append({"role": "assistant", "content": text})
        self.turns.append(
            {
                "prompt_tokens": prompt_tokens,
                "gen_tokens": int(gen_ids.shape[0]),
                "latency_s": round(elapsed, 2),
                "rep_rate": round(ngram_rep_rate(text), 3),
                "length": len(text.strip()),
                "text": text,
            }
        )
        return text


class EvalModel:
    def __init__(self, name: str, weight: Path, official: bool, device: str):
        self.name = name
        self.official = official
        self.device = device
        from transformers import AutoTokenizer

        from minimind_reborn import envs

        self.tokenizer = AutoTokenizer.from_pretrained(envs.tokenizer_path())
        if official:
            import sys

            ref = REPO.parent.parent / "reference" / "minimind"
            if str(ref) not in sys.path:
                sys.path.insert(0, str(ref))
            from model.model_minimind import MiniMindConfig as OffCfg
            from model.model_minimind import MiniMindForCausalLM as OffLM

            self.model = OffLM(OffCfg(hidden_size=768, num_hidden_layers=8)).half().eval().to(device)
            self.model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
        else:
            from minimind_reborn.models.config import MiniMindConfig
            from minimind_reborn.models.model import MiniMindForCausalLM
            from minimind_reborn.models.weights import load_inference_weights

            self.model = MiniMindForCausalLM(MiniMindConfig()).half().eval().to(device)
            load_inference_weights(self.model, weight, strict=True)

    def new_session(self) -> MultiTurnSession:
        return MultiTurnSession(self.model, self.tokenizer, official=self.official)


# ===== Suite A：跨轮记忆与指代 =====
# (轮次剧本, 召回关键词列表)——首轮注入，末轮追问
A_CASES = [
    {
        "script": [
            "你好！我叫李小明，今年28岁，是一名程序员，最喜欢的编程语言是 Rust。",
            "我平时喜欢爬山，养了一只叫花花的猫。",
            "请根据前面我告诉你的信息，介绍一下我自己。",
        ],
        "recall_keywords": ["李小明", "28", "程序员", "Rust", "爬山", "花花"],
        "recall_turn": 3,
    },
    {
        "script": [
            "我们公司团建订了 17 份盒饭，每份 35 元。",
            "另外还租了一辆大巴，花了 900 元。",
            "请问这次团建总共花了多少钱？（需要的话请把两笔费用相加）",
        ],
        "recall_keywords": ["17", "35", "900"],
        "recall_turn": 3,
    },
    {
        "script": [
            "记住这个暗号：蓝天白云。",
            "暗号是什么？",
        ],
        "recall_keywords": ["蓝天白云"],
        "recall_turn": 2,
    },
    {
        "script": [
            "我最喜欢的一本书是《三体》，作者是刘慈欣。",
            "今天天气真不错。",
            "我最喜欢的那本书叫什么名字？作者是谁？",
        ],
        "recall_keywords": ["三体", "刘慈欣"],
        "recall_turn": 3,
    },
]


def suite_a(em: EvalModel) -> dict:
    results = {"per_case": [], "recall_hits": 0, "recall_total": 0}
    sample_lines = [f"## Suite A 跨轮记忆 — {em.name}", ""]
    for ci, case in enumerate(A_CASES):
        session = em.new_session()
        answers = []
        for i, utter in enumerate(case["script"]):
            ans = session.respond(utter)
            answers.append(ans)
            if i + 1 == case["recall_turn"]:
                recall_text = ans
        hits = [kw for kw in case["recall_keywords"] if kw in recall_text]
        results["recall_hits"] += len(hits)
        results["recall_total"] += len(case["recall_keywords"])
        results["per_case"].append(
            {"case": ci, "hits": hits, "missed": [k for k in case["recall_keywords"] if k not in hits]}
        )
        sample_lines += [
            f"### Case {ci}（召回 {len(hits)}/{len(case['recall_keywords'])}：命中 {hits or '无'}）",
            "```text",
        ]
        for i, (u, a) in enumerate(zip(case["script"], answers, strict=False)):
            sample_lines += [f"[用户 {i + 1}] {u}", f"[模型 {i + 1}] {a[:220]}", ""]
        sample_lines += ["```", ""]
    results["recall_rate"] = round(results["recall_hits"] / max(results["recall_total"], 1), 3)
    (SAMPLES_DIR / f"suiteA_{em.name.replace(' ', '_')}.md").write_text("\n".join(sample_lines), encoding="utf-8")
    return results


# ===== Suite B：上下文长度压力 =====
FILLER_FACTS = [
    "第{n}个知识点：水的沸点在一标准大气压下是100摄氏度。",
    "第{n}个知识点：光在真空中的速度约为每秒30万公里。",
    "第{n}个知识点：地球绕太阳公转一周约需365天。",
    "第{n}个知识点：铁的化学符号是Fe。",
    "第{n}个知识点：世界上最高的山峰是珠穆朗玛峰。",
    "第{n}个知识点：月球绕地球一圈约27天。",
]
FIXED_QUESTION = "请用一句话总结上面对话里出现过的知识点数量。"


def build_filler_history(target_tokens: int, tokenizer) -> list[dict[str, str]]:
    """构造接近 target_tokens 的合成多轮历史（user/assistant 交替的伪知识点问答）。"""
    history: list[dict[str, str]] = [
        {"role": "user", "content": "我们来玩一个知识积累游戏，我会陆续告诉你一些知识点。"}
    ]
    history.append({"role": "assistant", "content": "好的，请开始，我会记住你说的内容。"})
    n = 0
    while True:
        n += 1
        history.append({"role": "user", "content": FILLER_FACTS[(n - 1) % len(FILLER_FACTS)].format(n=n)})
        rendered = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        if len(tokenizer(rendered).input_ids) >= target_tokens:
            history.pop()  # 超了就丢掉最后一个 user，保持模板可生成
            break
        history.append({"role": "assistant", "content": f"已记录第{n}个知识点。"})
    return history


def suite_b(em: EvalModel) -> dict:
    results = {"levels": []}
    sample_lines = [f"## Suite B 上下文长度压力 — {em.name}", ""]
    for target in (256, 512, 768, 1024, 1536, 2048):
        session = em.new_session()
        session.history = build_filler_history(target, em.tokenizer)
        ans = session.respond(FIXED_QUESTION)
        turn = session.turns[-1]
        rep = ngram_rep_rate(ans)
        # 跑题判定：回答与"知识点数量"无关且未复述任何知识点编号
        on_topic = bool(re.search(r"\d+\s*个|知识点", ans))
        results["levels"].append(
            {
                "target_tokens": target,
                "actual_prompt_tokens": turn["prompt_tokens"],
                "gen_tokens": turn["gen_tokens"],
                "rep_rate": rep,
                "length": turn["length"],
                "on_topic": on_topic,
            }
        )
        sample_lines += [
            f"### target≈{target} tok（实际 prompt {turn['prompt_tokens']}）| 重复率 {rep:.3f} | {'在题' if on_topic else '跑题'}",
            "```text",
            ans[:200],
            "```",
            "",
        ]
    (SAMPLES_DIR / f"suiteB_{em.name.replace(' ', '_')}.md").write_text("\n".join(sample_lines), encoding="utf-8")
    return results


# ===== Suite C：长程多轮稳定性 =====
C_QUESTIONS = [
    "用一句话介绍什么是机器学习。",
    "那深度学习和它是什么关系？",
    "举一个生活中的应用例子。",
    "这个例子用到了什么技术？",
    "刚开始学这个领域应该看什么？",
    "需要数学基础吗？大概要哪些？",
    "线性代数里最核心的概念是什么？",
    "矩阵乘法有什么直观理解方式？",
    "回到第一个问题，再总结一次。",
    "你刚才一共回答了我几个问题？",
    "帮我把这些内容列一个学习路线。",
    "最后用一句话鼓励一下初学者。",
]


def suite_c(em: EvalModel) -> dict:
    session = em.new_session()
    sample_lines = [f"## Suite C 长程稳定性（12 轮）— {em.name}", ""]
    for i, q in enumerate(C_QUESTIONS, 1):
        ans = session.respond(q)
        t = session.turns[-1]
        sample_lines += [
            f"### 轮 {i}（prompt {t['prompt_tokens']} tok | 生成 {t['gen_tokens']} | 重复率 {t['rep_rate']:.3f}）",
            f"**Q:** {q}",
            f"**A:** {ans[:260]}",
            "",
        ]
    results = {
        "turns": session.turns,
        # 角色漂移粗判：末段出现引导性 user 口吻且无句号收尾
        "role_drift": any("user" in t["text"][-60:] and "。" not in t["text"][-60:] for t in session.turns),
        "avg_rep_last3": round(statistics.mean(t["rep_rate"] for t in session.turns[-3:]), 3),
        "avg_rep_first3": round(statistics.mean(t["rep_rate"] for t in session.turns[:3]), 3),
    }
    (SAMPLES_DIR / f"suiteC_{em.name.replace(' ', '_')}.md").write_text("\n".join(sample_lines), encoding="utf-8")
    return results


# ===== Suite D：长度边界行为 =====
def suite_d(em: EvalModel) -> dict:
    """超 32768 token 的 prompt：验证 RoPE 表越界的实际表现（守卫缺口）。"""
    result = {"behavior": "n/a", "error_head": ""}
    session = em.new_session()
    session.history.append({"role": "user", "content": "我们来测试超长上下文。"})
    session.history.append({"role": "assistant", "content": "好的。"})
    filler = build_filler_history(40000, em.tokenizer)
    session.history = filler
    try:
        session.respond("总结一下。")
        result["behavior"] = "no-error (unexpected)"
    except RuntimeError as e:
        result["behavior"] = "RuntimeError (shape mismatch expected)"
        result["error_head"] = str(e)[:160]
    except Exception as e:  # noqa: BLE001
        result["behavior"] = f"OtherError: {type(e).__name__}"
        result["error_head"] = str(e)[:160]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="多轮/连续对话能力检测")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=str(REPO / "out" / "multiturn_eval.md"))
    parser.add_argument("--suites", default="ABCD", help="要跑的套件（默认 ABCD）")
    args = parser.parse_args()
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    weight_specs = [
        ("ours_dpo", REPO / "out" / "dpo_768.pth", False),
        ("ours_full_sft", REPO / "out" / "full_sft_768.pth", False),
        ("official_full_sft", OFFICIAL_DIR / "full_sft_768.pth", True),
    ]

    report: list[str] = ["# 多轮/连续对话能力检测", ""]

    for name, weight, official in weight_specs:
        print(f"===== {name} =====")
        em = EvalModel(name, weight, official, args.device)
        report += [f"## {name}", ""]

        if "A" in args.suites:
            r = suite_a(em)
            report += [
                f"**Suite A 跨轮记忆召回率：{r['recall_rate']:.1%}**（命中 {r['recall_hits']}/{r['recall_total']}）",
                "",
            ]
            print(f"  A 召回率 {r['recall_rate']:.1%}")

        if "B" in args.suites:
            r = suite_b(em)
            report += [
                "**Suite B 上下文长度压力**",
                "",
                "| 目标长度 | 实际 prompt tok | 生成 tok | 重复率 | 在题 |",
                "|---|---|---|---|---|",
            ]
            for lv in r["levels"]:
                report.append(
                    f"| ≈{lv['target_tokens']} | {lv['actual_prompt_tokens']} | {lv['gen_tokens']} | {lv['rep_rate']:.3f} | {'✅' if lv['on_topic'] else '❌'} |"
                )
            report += [""]
            print("  B 完成")

        if "C" in args.suites:
            r = suite_c(em)
            report += [
                f"**Suite C 长程稳定性（12 轮）**：前 3 轮平均重复率 {r['avg_rep_first3']} → 末 3 轮 {r['avg_rep_last3']}；角色漂移 {'是 ⚠' if r['role_drift'] else '否'}",
                "",
            ]
            print(f"  C 前3/末3 重复率 {r['avg_rep_first3']}/{r['avg_rep_last3']}")

        if "D" in args.suites and name == "ours_dpo":  # 边界行为只需测一个模型
            r = suite_d(em)
            report += [
                f"**Suite D 边界行为（>32768 tok prompt）**：{r['behavior']}",
                "",
                f"> 错误摘要：`{r['error_head']}`",
                "",
            ]
            print(f"  D {r['behavior']}")

        del em
        torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\n已追加到 {out_path}")


if __name__ == "__main__":
    main()

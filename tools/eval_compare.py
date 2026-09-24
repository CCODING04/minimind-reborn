"""质量评测（Phase C）：5 权重横评——量化"欠缺与不足"。

权重集：我们 pretrain_768(base 直拼) / full_sft_768 / dpo_768 + 官方 pretrain_768 / full_sft_768。
维度：
- 生成质量：4 组 prompt（连贯性 A / 格式延续 B / 风格模仿 C / 长文本记忆 D）× 3 seed；
  量化 3-gram 重复率（字符级）、平均生成长度、eos 命中率；
- 损失：相同评测语料（pretrain val 切片 20 batch + sft val 切片 10 batch）上的 CE / ppl。
  注意混杂：官方权重在 full 10GB 数据上训练，我们只用 mini 1.2GB——损失差距主要反映数据量而非代码。

用法：uv run python tools/eval_compare.py --device cuda:1
产出：out/official_compare.md 追加"质量横评"节 + out/eval_samples/ 逐样例文本。
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
OFFICIAL_DIR = REPO / "out" / "official"
REFERENCE = REPO.parent.parent / "reference" / "minimind"
OFFICIAL_SYS_PATH = str(REFERENCE)

GROUPS: dict[str, list[str]] = {
    "A_连贯性": [
        "人工智能是",
        "春天的公园里，",
        "中国的四大发明包括",
    ],
    "B_格式延续": [
        "以下是几种常见的编程语言：1. Python 2.",
        " shopping list:\n- apples\n- milk\n-",
        "会议纪要\n时间：周一上午\n参会人：张三、李四\n议题：",
    ],
    "C_风格模仿": [
        "从前有座山，山里有座庙，庙里有个老和尚。有一天，",
        "在遥远的银河系边缘，有一颗孤独的行星。那里的居民",
        "月光如水，流泻在青石板路上。",
    ],
    "D_长文本记忆": [
        "小李是一名28岁的程序员，在北京工作。他每天早上七点起床，坐地铁去公司，晚上经常加班到九点。他养了一只叫花花的猫，"
        "周末喜欢去爬山。请总结一下小李的生活：",
    ],
}
SEEDS = [42, 43, 44]


def ngram_rep_rate(text: str, n: int = 3) -> float:
    """字符级 n-gram 重复率：1 - 去重后/总数（0=无重复，1=完全循环）。"""
    cleaned = text.replace(" ", "").replace("\n", "")
    if len(cleaned) < n + 1:
        return 0.0
    grams = [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


class EvalModel:
    """统一生成接口：内部区分我们的模型（走 inference.generator）与官方模型（走原版 generate）。"""

    def __init__(self, name: str, weight: Path, *, pretrain_mode: bool, official: bool, device: str):
        self.name = name
        self.pretrain_mode = pretrain_mode
        self.official = official
        self.device = device
        from transformers import AutoTokenizer

        from minimind_reborn import envs

        self.tokenizer = AutoTokenizer.from_pretrained(envs.tokenizer_path())
        if official:
            if OFFICIAL_SYS_PATH not in sys.path:
                sys.path.insert(0, OFFICIAL_SYS_PATH)
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

    def generate(self, prompt: str, *, seed: int, max_new_tokens: int = 128) -> str:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if self.pretrain_mode:
            text = self.tokenizer.bos_token + prompt
        else:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        from minimind_reborn.configuration.schemas import GenerateConfig

        with torch.inference_mode():
            if self.official:
                out = self.model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                ids = out[0][inputs["input_ids"].shape[1] :]
            else:
                from minimind_reborn.inference.generator import generate

                out = generate(
                    self.model,
                    inputs["input_ids"],
                    GenerateConfig(**_GEN_KW),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                )
                ids = out["generated_ids"][0]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        # 停止三重之 3：截断 eos 后残留
        eos = self.tokenizer.eos_token_id
        return (
            text.split(self.tokenizer.eos_token)[0] if eos is not None and self.tokenizer.eos_token in text else text
        )


_GEN_KW = {"temperature": 0.8, "top_p": 0.9, "top_k": 50, "max_new_tokens": 128}  # 与官方横评口径一致


@torch.inference_mode()
def eval_ce_loss(weight: Path, *, official: bool, device: str, batches: int = 20) -> dict:
    """在固定评测语料上算 CE / ppl（pretrain val 切片 20 batch + sft val 切片 10 batch）。"""
    from transformers import AutoTokenizer

    from minimind_reborn import envs
    from minimind_reborn.data.datasets import PretrainDataset, SFTDataset
    from minimind_reborn.data.registry import file_path, resolve_dataset
    from minimind_reborn.models.config import MiniMindConfig
    from minimind_reborn.models.model import MiniMindForCausalLM
    from minimind_reborn.models.weights import load_inference_weights

    tokenizer = AutoTokenizer.from_pretrained(envs.tokenizer_path())
    if official:
        if OFFICIAL_SYS_PATH not in sys.path:
            sys.path.insert(0, OFFICIAL_SYS_PATH)
        from model.model_minimind import MiniMindConfig as OffCfg
        from model.model_minimind import MiniMindForCausalLM as OffLM

        model = OffLM(OffCfg(hidden_size=768, num_hidden_layers=8)).half().eval().to(device)
        model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
        forward = lambda m, ids, labels: m(ids, labels=labels).loss.item()  # noqa: E731
    else:
        model = MiniMindForCausalLM(MiniMindConfig()).half().eval().to(device)
        load_inference_weights(model, weight, strict=True)
        forward = lambda m, ids, labels: m(ids, labels=labels).loss.item()  # noqa: E731

    losses = []
    pre = PretrainDataset(file_path(resolve_dataset("pretrain_t2t_mini")), tokenizer, max_length=340, eval_ratio=0.005)
    ev = pre.eval_view()
    count = 0
    for i in range(min(batches, len(ev))):
        item = ev[i]
        ids = item["input_ids"].unsqueeze(0).to(device)
        labels = item["labels"].unsqueeze(0).to(device)
        losses.append(forward(model, ids, labels))
        count += 1
    sft = SFTDataset(file_path(resolve_dataset("sft_t2t_mini")), tokenizer, max_length=768, eval_ratio=0.005)
    ev2 = sft.eval_view()
    for i in range(min(10, len(ev2))):
        item = ev2[i]
        ids = item["input_ids"].unsqueeze(0).to(device)
        labels = item["labels"].unsqueeze(0).to(device)
        losses.append(forward(model, ids, labels))
        count += 1
    mean_ce = statistics.mean(losses)
    return {"ce": round(mean_ce, 4), "ppl": round(2.718281828**mean_ce, 2), "batches": count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--skip-loss", action="store_true", help="只跑生成横评（调试用）")
    args = parser.parse_args()

    weight_specs = [
        ("ours pretrain (base直拼)", REPO / "out" / "pretrain_768.pth", True, False),
        ("ours full_sft", REPO / "out" / "full_sft_768.pth", False, False),
        ("ours dpo", REPO / "out" / "dpo_768.pth", False, False),
        ("official pretrain (base直拼)", OFFICIAL_DIR / "pretrain_768.pth", True, True),
        ("official full_sft", OFFICIAL_DIR / "full_sft_768.pth", False, True),
    ]

    samples_dir = REPO / "out" / "eval_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    report: list[str] = ["", "## 质量横评（5 权重）", ""]
    quant_rows = []

    for name, weight, pretrain_mode, official in weight_specs:
        print(f"=== {name} ===")
        em = EvalModel(name, weight, pretrain_mode=pretrain_mode, official=official, device=args.device)
        rep_rates, lengths, eos_hits, total = [], [], 0, 0
        sample_lines = [f"## {name}", ""]
        for group, prompts in GROUPS.items():
            for prompt in prompts:
                for seed in SEEDS:
                    text = em.generate(prompt, seed=seed)
                    rep_rates.append(ngram_rep_rate(text))
                    lengths.append(len(text))
                    total += 1
                    if text.strip():
                        eos_hits += 1
                    if seed == SEEDS[0]:
                        sample_lines += [f"### {group} | {prompt[:24]}…", "```text", text[:400], "```", ""]
        quant_rows.append(
            f"| {name} | {statistics.mean(rep_rates):.3f} | {statistics.mean(lengths):.0f} | {eos_hits}/{total} |"
        )
        (samples_dir / f"{name.replace(' ', '_').replace('/', '_')}.md").write_text(
            "\n".join(sample_lines), encoding="utf-8"
        )
        print(
            f"  重复率 {statistics.mean(rep_rates):.3f} | "
            f"平均长度 {statistics.mean(lengths):.0f} | eos {eos_hits}/{total}"
        )
        del em
        torch.cuda.empty_cache()

    report += [
        "指标说明：3-gram 字符重复率（0=无循环，1=完全复读，越低越好）；",
        "平均生成长度（字符）；eos 命中 = 生成非空比例。",
        "采样：temperature 0.8 / top_p 0.9 / top_k 50，3 seed × 9-10 prompt，max_new_tokens 128。",
        "",
        "| 权重 | 3-gram 重复率↓ | 平均长度 | 非空生成 |",
        "|---|---|---|---|",
        *quant_rows,
    ]

    if not args.skip_loss:
        report += [
            "",
            "### 相同评测集上的损失对比",
            "",
            "评测语料：我们 pretrain val 切片 20 batch（seq 340）\n+ sft val 切片 10 batch（seq 768）。",
            "**混杂说明**：官方权重在 full 10GB 语料训练，我们只用 mini 1.2GB——"
            "损失差距主要反映数据量与训练步数，而非代码差异。",
            "",
            "| 权重 | CE loss↓ | ppl↓ | batches |",
            "|---|---|---|---|",
        ]
        for name, weight, _pretrain_mode, official in weight_specs:
            m = eval_ce_loss(weight, official=official, device=args.device)
            report.append(f"| {name} | {m['ce']} | {m['ppl']} | {m['batches']} |")
            print(f"  {name}: CE {m['ce']} ppl {m['ppl']}")
            del m
            torch.cuda.empty_cache()

    out_path = Path(args.out) if hasattr(args, "out") else REPO / "out" / "official_compare.md"
    with out_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\n已追加到 {out_path}")


if __name__ == "__main__":
    main()

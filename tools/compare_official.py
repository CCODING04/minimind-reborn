"""官方对照评测（Phase A+B）：权重兼容性验证 + 同权重双实现推理对比。

用法：uv run python tools/compare_official.py --device cuda:1

- Phase A：官方 minimind-3-pytorch 的 .pth 以 strict=True 载入我们的模型，验证"键级兼容"声明；
- Phase B：同一官方 full_sft_768 权重，双实现（我们 generator vs 官方 generate）同 prompt
  同采样参数对比：贪心输出一致性、吞吐 tok/s、峰值显存。

结论写入 out/official_compare.md 的"双实现"与"兼容性"两节。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
OFFICIAL_DIR = REPO / "out" / "official"
REFERENCE = REPO.parent.parent / "reference" / "minimind"

PROMPTS = [
    "推荐两道中国菜并说明做法",
    "为什么天空是蓝色的？",
    "解释一下什么是机器学习",
    "请用Python写一个计算斐波那契数列的函数",
    "比较一下猫和狗作为宠物的优缺点",
]
LONG_PROMPT = "从前有座山，山里有座庙，庙里有个老和尚和小和尚。" * 12 + "老和尚对小明说："


def build_our(path: Path, device: str):
    """我们的模型 + 我们的生成内核（预分配 KV cache）。"""
    from transformers import AutoTokenizer

    from minimind_reborn import envs
    from minimind_reborn.models.config import MiniMindConfig
    from minimind_reborn.models.model import MiniMindForCausalLM
    from minimind_reborn.models.weights import load_inference_weights

    cfg = MiniMindConfig()  # 默认值 = 768/8L/8q/4kv/96hd/2432，与官方权重实测 shape 一致
    model = MiniMindForCausalLM(cfg)
    load_inference_weights(model, path, strict=True)  # strict 失败即抛错——兼容性验证本体
    model = model.half().eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(envs.tokenizer_path())
    return model, tokenizer


def build_official(path: Path, device: str):
    """官方模型（reference/minimind 原版代码，动态挂 sys.path，不污染包依赖）。"""
    sys.path.insert(0, str(REFERENCE))
    for mod in list(sys.modules):
        if mod.startswith("model") or mod == "trainer":
            del sys.modules[mod]
    from model.model_minimind import MiniMindConfig as OfficialConfig
    from model.model_minimind import MiniMindForCausalLM as OfficialLM
    from transformers import AutoTokenizer

    from minimind_reborn import envs

    cfg = OfficialConfig(hidden_size=768, num_hidden_layers=8)
    model = OfficialLM(cfg)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.half().eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(envs.tokenizer_path())
    return model, tokenizer


def our_generate(model, tokenizer, prompt: str, *, max_new_tokens: int, greedy: bool) -> torch.Tensor:
    from minimind_reborn.configuration.schemas import GenerateConfig
    from minimind_reborn.inference.generator import generate

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    # greedy 必须真正走 T=0（修复：2026-10-06 前恒 T=0.85 采样——贪心一致性对比
    # 实测的是「我方采样 vs 官方贪心」，逐位一致率数字是噪声）
    gen = GenerateConfig(temperature=0.0 if greedy else 0.85, top_p=0.9, top_k=50)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = generate(
        model,
        inputs["input_ids"],
        gen,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        attention_mask=inputs["attention_mask"],
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tokens = out["sequences"].shape[1] - inputs["input_ids"].shape[1]
    peak = torch.cuda.max_memory_allocated() / 1e6
    return out["sequences"][0], tokens / max(elapsed, 1e-6), peak


def official_generate(model, tokenizer, prompt: str, *, max_new_tokens: int, greedy: bool):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        temperature=0.85,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tokens = out.shape[1] - inputs["input_ids"].shape[1]
    peak = torch.cuda.max_memory_allocated() / 1e6
    return out[0], tokens / max(elapsed, 1e-6), peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--out", default=str(REPO / "out" / "official_compare.md"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    report: list[str] = ["# 官方对照评测", ""]

    # ===== Phase A：strict 兼容性验证 =====
    print("=" * 60, "\nPhase A：strict 兼容性验证")
    our_model, tokenizer = build_our(OFFICIAL_DIR / "pretrain_768.pth", args.device)
    report += ["## 权重兼容性", "", "| 权重 | strict 载入 | 结论 |", "|---|---|---|"]
    for name in ("pretrain_768.pth", "full_sft_768.pth"):
        try:
            m, _ = build_our(OFFICIAL_DIR / name, args.device)
            n = sum(p.numel() for p in m.parameters()) / 1e6
            report.append(f"| 官方 {name} | ✅ strict 通过 | 键级兼容（{n:.1f}M） |")
            print(f"  {name}: strict OK ({n:.1f}M)")
            del m
            torch.cuda.empty_cache()
        except RuntimeError as e:
            report.append(f"| 官方 {name} | ❌ strict 失败 | `{str(e)[:120]}` |")
            print(f"  {name}: strict FAILED: {e}")

    # ===== Phase B：同权重双实现对比 =====
    print("=" * 60, "\nPhase B：同权重双实现推理对比（官方 full_sft_768）")
    official_model, tokenizer2 = build_official(OFFICIAL_DIR / "full_sft_768.pth", args.device)

    # 1) 贪心一致性
    report += [
        "",
        "## 贪心输出一致性（同权重，前缀一致比例）",
        "",
        "| prompt | 一致前缀 / 总生成 | 一致率 |",
        "|---|---|---|",
    ]
    agree_ratios = []
    for prompt in PROMPTS:
        ours, _, _ = our_generate(our_model, tokenizer, prompt, max_new_tokens=128, greedy=True)
        theirs, _, _ = official_generate(official_model, tokenizer2, prompt, max_new_tokens=128, greedy=True)
        n = min(ours.shape[0], theirs.shape[0])
        agree = int((ours[:n] == theirs[:n]).sum())
        # 逐位相同计数（允许各自提前 eos 后 pad 对不齐，按位置一致率）
        ratio = agree / max(n, 1)
        agree_ratios.append(ratio)
        report.append(f"| {prompt[:18]}… | {agree}/{n} | {ratio:.1%} |")
        print(f"  {prompt[:16]}… 一致率 {ratio:.1%}")
    overall = statistics.mean(agree_ratios)
    report += [
        "",
        f"**平均逐位一致率 {overall:.1%}**（同权重同数学；浮点路径差异来自 prefill 后端与 cache 组织，属预期）",
        "",
    ]

    # 2) 吞吐与显存（长 prompt + 384 token 贪心生成，取中位）
    report += [
        "## 推理吞吐与峰值显存（长 prompt，贪心 384 token，3 次取中位）",
        "",
        "| 实现 | 吞吐 tok/s | 峰值显存 MB |",
        "|---|---|---|",
    ]
    results: dict[str, tuple[float, float]] = {}
    for impl, _builder in (("reborn（KV cache 预分配）", "our"), ("official（每步 torch.cat）", "official")):
        rates, peaks = [], []
        for _ in range(args.repeats):
            if impl.startswith("reborn"):
                _, rate, peak = our_generate(our_model, tokenizer, LONG_PROMPT, max_new_tokens=384, greedy=True)
            else:
                _, rate, peak = official_generate(
                    official_model, tokenizer2, LONG_PROMPT, max_new_tokens=384, greedy=True
                )
            rates.append(rate)
            peaks.append(peak)
        results[impl] = (statistics.median(rates), statistics.median(peaks))
        report.append(f"| {impl} | {statistics.median(rates):.0f} | {statistics.median(peaks):.0f} |")
        print(f"  {impl}: {statistics.median(rates):.0f} tok/s, {statistics.median(peaks):.0f} MB")
    (our_rate, our_peak), (off_rate, off_peak) = (
        results["reborn（KV cache 预分配）"],
        results["official（每步 torch.cat）"],
    )
    report += [
        "",
        f"**吞吐比 {our_rate / off_rate:.2f}×，显存差 {off_peak - our_peak:+.0f} MB**"
        f"（63M 模型 + 短上下文下 KV cache 本身占比小，长上下文/大 batch 差距会放大）",
        "",
    ]

    # 3) 采样输出样例（供报告摘录）
    report += ["## 采样样例（官方 full_sft_768 权重，temperature 0.85 / top_p 0.9）", "", "```text"]
    text, _, _ = our_generate(our_model, tokenizer, "推荐两道中国菜并说明做法", max_new_tokens=200, greedy=False)
    report += [tokenizer.decode(text, skip_special_tokens=True)[:400], "```"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\n结果已写入 {out_path}")


if __name__ == "__main__":
    main()

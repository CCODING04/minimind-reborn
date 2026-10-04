"""长上下文续训子集构建（重训计划档二 4a）。

从 pretrain_t2t.jsonl 筛长文档（≥768 token，即 seq 380 时代被截断丢弃 45% 的那批）
+ 混入短文档回放防遗忘，产出 pretrain_longctx.jsonl 并注册。

- 阈值校准：先抽 500 行 tokenize 拟合「字符→token」比率，把 768 token 换算为字符阈值
  （精度要求 ≥90%：字符筛只做粗召回，token 精筛在换算后逐行 tokenize 复核太慢——
  采用比率阈值 + 抽样验证报告精度，不逐行精筛）
- 组成：长文档 ~75% + 短文档回放 ~25%（seed 42 确定性）
- 预算：--budget-chars 控制长文档总量（默认对应 ~0.8B token）

用法：uv run python tools/build_longctx_subset.py [--long-min-tokens 768] [--budget-chars 1450000000]
"""

from __future__ import annotations

import argparse
import json
import random

from minimind_reborn import envs

SAMPLE_N = 500
SEED = 42


def calibrate_chars_per_token(tokenizer, path) -> float:
    """抽样拟合平均每 token 的字符数（中文主导语料 ≈1.0，混英文略高）。"""
    rng = random.Random(SEED)
    lines: list[bytes] = []
    with open(path, "rb") as f:
        pool = []
        for i, line in enumerate(f):
            if i < 20000:
                pool.append(line)
        lines = rng.sample(pool, SAMPLE_N)
    ratios = []
    for line in lines:
        text = str(json.loads(line)["text"])
        n_tok = len(tokenizer(text, add_special_tokens=False).input_ids)
        if n_tok >= 8:
            ratios.append(len(text) / n_tok)
    ratios.sort()
    return ratios[len(ratios) // 2]  # 中位数抗离群


def main() -> None:
    parser = argparse.ArgumentParser(description="构建长上下文续训子集")
    parser.add_argument("--long-min-tokens", type=int, default=768)
    parser.add_argument("--budget-chars", type=int, default=1_450_000_000, help="长文档总字符预算（≈0.8B token）")
    parser.add_argument("--replay-frac", type=float, default=0.25, help="短文档回放占产出行数比例")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(envs.tokenizer_path())
    src = envs.data_root() / "pretrain_t2t.jsonl"
    out = envs.data_root() / "pretrain_longctx.jsonl"

    cpt = calibrate_chars_per_token(tokenizer, src)
    # 两段式（首版单段字符筛实测精度仅 56%——字符与 token 的换算散布远大于中位比率）：
    # 宽松字符粗筛（0.7×，保召回）建候选池 → 池内逐行 tokenize 精筛（池只有百万级，可承受）
    coarse_chars = int(args.long_min_tokens * cpt * 0.7)
    print(f"校准：中位 {cpt:.2f} 字符/token；粗筛阈值 {coarse_chars} 字符（0.7× 保召回）")

    rng = random.Random(SEED)
    candidates: list[str] = []  # (text, token_len)
    shorts: list[str] = []
    n_total = 0
    with open(src, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            n_total += 1
            text = str(json.loads(line)["text"])
            if len(text) >= coarse_chars:
                candidates.append(text)
            elif rng.random() < 0.04:  # 短文档池：低比例预采样，末尾按配比精裁
                shorts.append(text)

    print(f"粗筛候选 {len(candidates)} 行，池内 tokenize 精筛 …")
    longs: list[str] = []
    long_chars = 0
    for text in candidates:
        if long_chars >= args.budget_chars:
            break
        n_tok = len(tokenizer(text, add_special_tokens=False).input_ids)
        if n_tok >= args.long_min_tokens:
            longs.append(json.dumps({"text": text}, ensure_ascii=False))
            long_chars += len(text)

    # 回放配比：短文档行数 = longs / (1-frac) * frac
    n_short_target = int(len(longs) / (1 - args.replay_frac) * args.replay_frac)
    rng.shuffle(shorts)
    shorts = shorts[:n_short_target]
    shorts = [json.dumps({"text": t}, ensure_ascii=False) for t in shorts]
    rows = longs + shorts
    rng.shuffle(rows)

    tmp = out.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(r + "\n")
    tmp.rename(out)

    # 精度复核：精筛后的长行按定义 ≥ long_min_tokens（by construction），抽查防呆
    checked = rng.sample(longs, min(300, len(longs)))
    hit = sum(
        1
        for r in checked
        if len(tokenizer(json.loads(r)["text"], add_special_tokens=False).input_ids) >= args.long_min_tokens
    )
    precision = hit / max(len(checked), 1)
    print(
        json.dumps(
            {
                "source_lines": n_total,
                "coarse_candidates": len(candidates),
                "long_selected": len(longs),
                "long_chars": long_chars,
                "replay_short": len(shorts),
                "output_rows": len(rows),
                "output": str(out),
                "coarse_char_threshold": coarse_chars,
                "long_precision_check": round(precision, 4),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if precision < 0.99:
        raise SystemExit(f"精筛精度异常（{precision:.1%} < 99%）：by construction 应≈100%，检查实现")


if __name__ == "__main__":
    main()

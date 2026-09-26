"""数据集画像工具（data engineering 的基础测量）。

对注册表数据集做采样画像：字段结构、长度分布、精确/近重复率、语言构成、
顺序性（训练/验证尾切分的分布风险）、SFT 轮次分布与回复重复率。

用法：
  uv run python tools/data_profile.py --name pretrain_t2t --sample 100000
  uv run python tools/data_profile.py --name sft_t2t --sample 50000
输出 JSON 到 stdout（--out 可另存文件）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

CJK = re.compile(r"[\u4e00-\u9fff]")
ASCII_LETTER = re.compile(r"[a-zA-Z]")


def normalize(text: str) -> str:
    """近重复判定的规范化：小写 + 去空白与标点。"""
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]", "", text.lower())


def ngram_rep(text: str, n: int = 3) -> float:
    cleaned = re.sub(r"\s", "", text)
    if len(cleaned) < n + 1:
        return 0.0
    grams = [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


def char_stats(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {}
    s = sorted(lengths)
    return {
        "n": len(s),
        "p10": s[int(0.10 * len(s))],
        "p50": s[len(s) // 2],
        "p90": s[int(0.90 * len(s))],
        "p99": s[min(int(0.99 * len(s)), len(s) - 1)],
        "max": s[-1],
        "mean": round(statistics.mean(s), 1),
    }


def lang_bucket(text: str) -> str:
    n = max(len(text), 1)
    cjk = len(CJK.findall(text)) / n
    ascii_ = len(ASCII_LETTER.findall(text)) / n
    if cjk > 0.15:
        return "zh-dominant" if cjk > ascii_ else "mixed"
    if ascii_ > 0.3:
        return "en-dominant"
    return "other"


def profile_pretrain(lines: list[dict]) -> dict[str, Any]:
    texts = [str(x.get("text", "")) for x in lines]
    lengths = [len(t) for t in texts]
    norm_hashes = [hashlib.md5(normalize(t).encode()).hexdigest() for t in texts if len(t) > 50]
    raw_hashes = [hashlib.md5(t.encode()).hexdigest() for t in texts if len(t) > 50]
    langs = Counter(lang_bucket(t) for t in texts)
    # 顺序性：相邻样本规范化后前 64 字符相同（类别聚簇证据，尾切片 val 的分布风险）
    consec_same = sum(
        1
        for i in range(1, len(texts))
        if normalize(texts[i])[:64] == normalize(texts[i - 1])[:64] and len(texts[i]) > 64
    )
    prefix_dup = len(texts) - len({normalize(t)[:64] for t in texts if len(t) > 64})
    return {
        "kind": "pretrain",
        "fields": sorted({k for x in lines for k in x}),
        "char_len": char_stats(lengths),
        "exact_dup_rate_sample": round(1 - len(set(raw_hashes)) / len(raw_hashes), 4) if raw_hashes else 0,
        "normalized_dup_rate_sample": round(1 - len(set(norm_hashes)) / len(norm_hashes), 4) if norm_hashes else 0,
        "prefix64_dup_rate_sample": round(prefix_dup / max(len(texts), 1), 4),
        "consecutive_prefix_same_rate": round(consec_same / max(len(texts) - 1, 1), 4),
        "lang_mix": dict(langs),
    }


def profile_sft(lines: list[dict]) -> dict[str, Any]:
    user_lens, resp_lens, resp_reps = [], [], []
    user_prefixes, resp_norm_hashes = [], []
    langs: Counter[str] = Counter()
    turns_hist: Counter[int] = Counter()
    for x in lines:
        convs = x.get("conversations", [])
        turns_hist[len(convs)] += 1
        for c in convs:
            if c.get("role") == "assistant":
                text = str(c.get("content", ""))
                resp_lens.append(len(text))
                resp_norm_hashes.append(hashlib.md5(normalize(text).encode()).hexdigest())
                resp_reps.append(ngram_rep(text))
                langs[lang_bucket(text)] += 1
            elif c.get("role") == "user":
                u = str(c.get("content", ""))
                user_lens.append(len(u))
                user_prefixes.append(normalize(u)[:64])
    return {
        "kind": "sft",
        "fields": sorted({k for x in lines for k in x}),
        "turns_distribution": {str(k): v for k, v in sorted(turns_hist.items())},
        "multi_turn_rate": round(sum(v for k, v in turns_hist.items() if k > 2) / max(sum(turns_hist.values()), 1), 4),
        "user_char_len": char_stats(user_lens),
        "response_char_len": char_stats(resp_lens),
        "response_exact_dup_rate": round(1 - len(set(resp_norm_hashes)) / len(resp_norm_hashes), 4)
        if resp_norm_hashes
        else 0,
        "user_prefix64_dup_rate": round(1 - len(set(user_prefixes)) / len(user_prefixes), 4) if user_prefixes else 0,
        "response_3gram_rep_mean": round(statistics.mean(resp_reps), 4) if resp_reps else 0,
        "response_lang_mix": dict(langs),
    }


def profile_dpo(lines: list[dict]) -> dict[str, Any]:
    lens_chosen, lens_rejected = [], []
    for x in lines:
        ch = x.get("chosen", [])
        rj = x.get("rejected", [])
        lens_chosen.append(len(str(ch[-1].get("content", ""))) if ch else 0)
        lens_rejected.append(len(str(rj[-1].get("content", ""))) if rj else 0)
    return {
        "kind": "dpo",
        "fields": sorted({k for x in lines for k in x}),
        "chosen_char_len": char_stats(lens_chosen),
        "rejected_char_len": char_stats(lens_rejected),
        "chosen_longer_rate": round(
            sum(1 for c, r in zip(lens_chosen, lens_rejected, strict=False) if c > r) / max(len(lens_chosen), 1), 3
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="数据集画像")
    parser.add_argument("--name", required=True, help="注册表数据集名")
    parser.add_argument("--sample", type=int, default=50000, help="采样行数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from minimind_reborn.data.datasets import JsonlIndexedDataset
    from minimind_reborn.data.registry import file_path, resolve_dataset

    entry = resolve_dataset(args.name)
    path = file_path(entry)
    ds = JsonlIndexedDataset(path)
    rng = random.Random(args.seed)
    idx = sorted(rng.sample(range(len(ds)), min(args.sample, len(ds))))

    lines = [ds.load_line(i) for i in idx]

    if entry.kind == "pretrain":
        profile: dict[str, Any] = profile_pretrain(lines)
    elif entry.kind == "sft":
        profile = profile_sft(lines)
    elif entry.kind == "dpo":
        profile = profile_dpo(lines)
    else:
        profile = {"kind": entry.kind, "note": "RL 数据集画像暂未实现"}

    result = {
        "dataset": args.name,
        "file": str(path),
        "file_bytes": path.stat().st_size,
        "total_lines": len(ds),
        "sampled": len(lines),
        "sample_seed": args.seed,
        **profile,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

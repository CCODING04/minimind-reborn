"""token 级长度画像：长上下文维度的截断损毁测量（与 data_profile.py 的字符级互补）。

测量内容（采样 seed 42，字节偏移随机采样避免头部偏差）：
- pretrain：token 长度分布、≥训练截断(380) 占比、截断保留率（实际进入训练的 token 比例）、长样本潜力
- sft：渲染后会话 token 分布、≥768 占比、多轮样本截断损毁
  （assistant 完整回复区间存活率 / loss-mask 全零样本率 / 半截残段计数）
- dpo：chosen/rejected ≥1024 占比（配对级）

用法：
  uv run python tools/len_profile.py --name pretrain_t2t --sample 30000
  uv run python tools/len_profile.py --name sft_t2t --sample 30000
输出 JSON 到 stdout。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any

TOK: Any = None  # 惰性初始化（main 里加载）
PRE_IDS: list[int] = []
SUF_IDS: list[int] = []
LPRE = LSUF = 0
SEED = 42


def init_tokenizer() -> None:
    global TOK, PRE_IDS, SUF_IDS, LPRE, LSUF
    if TOK is not None:
        return
    from transformers import AutoTokenizer

    from minimind_reborn import envs
    from minimind_reborn.data.loss_mask import pattern_ids

    TOK = AutoTokenizer.from_pretrained(envs.tokenizer_path())
    PRE_IDS, SUF_IDS = pattern_ids(TOK)
    LPRE, LSUF = len(PRE_IDS), len(SUF_IDS)


def sample_lines(path: str | Path, n: int, seed: int = SEED) -> list[bytes]:
    """字节偏移随机采样完整行（避免头部偏差）。"""
    rng = random.Random(seed)
    size = os.path.getsize(path)
    seen: set[int] = set()
    out: list[bytes] = []

    def batch(k: int) -> None:
        offsets = sorted(rng.sample(range(size), k))
        with open(path, "rb") as f:
            for pos in offsets:
                f.seek(pos)
                f.readline()  # 丢弃残行
                start = f.tell()
                line = f.readline()
                if not line or not line.strip() or start in seen:
                    continue
                seen.add(start)
                out.append(line)

    guard = 0
    while len(out) < n and guard < 10:
        guard += 1
        batch(int((n - len(out)) * 1.15) + 16)
    return out[:n]


def pctl(s: list[float], p: float) -> float:
    s = sorted(s)
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    return float(s[lo]) if lo == hi else s[lo] * (hi - idx) + s[hi] * (idx - lo)


def describe(vals: list[float]) -> dict[str, float]:
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 1),
        "p10": round(pctl(vals, 10)),
        "p50": round(pctl(vals, 50)),
        "p90": round(pctl(vals, 90)),
        "p99": round(pctl(vals, 99)),
        "max": int(max(vals)),
    }


def frac_ge(vals: list[float], t: int) -> float:
    return round(sum(1 for v in vals if v >= t) / len(vals), 4)


def scan_spans(ids: list[int]) -> tuple[int, int]:
    """扫描 assistant 完整区间与残段——已与训练同源（loss_mask.scan_response_spans）。

    C2 语义：残段（有 <|im_start|>assistant\\n 开头但 <|im_end|>\\n 被截断丢失）
    不计入监督区间，仅计数供诊断。
    """
    from minimind_reborn.data.loss_mask import scan_response_spans

    spans, dangling = scan_response_spans(ids, PRE_IDS, SUF_IDS)
    return len(spans), dangling


def render_conv(convs: list[dict]) -> str:
    """渲染单条对话（与 SFTDataset.render 同源，经 render_conversations；无随机 system 插入）。"""
    from minimind_reborn.data.datasets import render_conversations

    return render_conversations(TOK, convs, remove_empty_think=False)


def profile_pretrain_len(lines: list[bytes], train_seq: int) -> dict[str, Any]:
    lens = [len(TOK(json.loads(line)["text"], add_special_tokens=False).input_ids) for line in lines]
    kept = sum(min(v, train_seq - 2) for v in lens)  # -2 给 bos/eos 让位
    total = sum(lens)
    return {
        "token_len": describe([float(v) for v in lens]),
        "truncated_at_train_seq": frac_ge([float(v) for v in lens], train_seq),
        "truncation_retention_rate": round(kept / total, 4),
        "token_never_trained_rate": round(1 - kept / total, 4),
        "long_sample_potential": {
            "ge_1024": frac_ge([float(v) for v in lens], 1024),
            "ge_2048": frac_ge([float(v) for v in lens], 2048),
        },
    }


def profile_sft_len(lines: list[bytes], train_seq: int) -> dict[str, Any]:
    """SFT 截断损毁画像——与训练完全同源（encode_sft_sample：C1 轮次适配 + C2 残段不计损）。

    测量口径：full_len 为渲染全长（未适配）；survival = 适配后完整回复数 / 适配前完整回复数；
    dangling/hard_truncated/zero_loss 均为适配后的最终训练语义。
    """
    from minimind_reborn.data.datasets import encode_sft_sample

    conv_lens: list[float] = []
    multi_lens: list[float] = []
    survival: list[float] = []
    zero_mask = 0
    multi_n = 0
    damaged_n = 0  # full_len > train_seq 的多轮样本（旧口径下必然拦腰受损）
    fitted_n = 0  # 触发轮次适配
    hard_trunc_n = 0  # 单回复超窗硬截
    dangling_total = 0  # 残段总数（C2 后不计损，仅诊断）
    complete_after_total = 0
    for line in lines:
        convs = json.loads(line)["conversations"]
        out = encode_sft_sample(TOK, PRE_IDS, SUF_IDS, convs, train_seq, remove_empty_think=False)
        stats = out["stats"]
        conv_lens.append(float(stats["full_len"]))
        if len(convs) > 2:
            multi_n += 1
            multi_lens.append(float(stats["full_len"]))
            if stats["full_len"] > train_seq:
                damaged_n += 1
                if stats["fitted"]:
                    fitted_n += 1
                if stats["hard_truncated"]:
                    hard_trunc_n += 1
                dangling_total += stats["dangling"]
                complete_after_total += stats["complete_after"]
                if stats["complete_before"] > 0:
                    survival.append(stats["complete_after"] / stats["complete_before"])
                if stats["zero_loss"]:
                    zero_mask += 1
    kept_total_spans = complete_after_total + dangling_total
    return {
        "conv_token_len": describe(conv_lens),
        "truncated_at_train_seq": frac_ge(conv_lens, train_seq),
        "multi_turn": {
            "share_of_sample": round(multi_n / max(len(lines), 1), 4),
            "token_len": describe(multi_lens) if multi_lens else {},
            "truncated_at_train_seq": frac_ge(multi_lens, train_seq) if multi_lens else 0,
            "damaged_over_seq": damaged_n,
            "turn_fitted_rate": round(fitted_n / max(damaged_n, 1), 4),
            "hard_truncated_rate": round(hard_trunc_n / max(damaged_n, 1), 4),
            "dangling_per_damaged": round(dangling_total / max(damaged_n, 1), 4),
            # 结构保证主指标（§4.1）：留下的回复区间中完整区间的占比，目标 100%
            # （残段唯一来源是「单条回复自身超窗」的硬截，这类样本由 C3 在训练期过滤）
            "assistant_span_intact_rate": round(complete_after_total / max(kept_total_spans, 1), 4),
            # 旧口径（完整后/完整前）：轮次丢弃按设计减少分子（旧轮次回复被整轮舍弃），
            # 与旧拦腰截断的「完整存活率」语义不同，不可直比
            "assistant_span_survival_rate": round(statistics.mean(survival), 4) if survival else None,
            "zero_loss_mask_rate": round(zero_mask / max(multi_n, 1), 4) if multi_n else 0,
        },
    }


def _render_side(convs: list[dict]) -> str:  # pragma: no cover
    return TOK.apply_chat_template(convs, tokenize=False, add_generation_prompt=False)


def profile_dpo_len(lines: list[bytes], train_seq: int) -> dict[str, Any]:
    chosen: list[float] = []
    rejected: list[float] = []
    pair_trunc = 0
    for line in lines:
        d = json.loads(line)
        cl = len(TOK(_render_side(d["chosen"]), add_special_tokens=False).input_ids)
        rl = len(TOK(_render_side(d["rejected"]), add_special_tokens=False).input_ids)
        chosen.append(float(cl))
        rejected.append(float(rl))
        if cl > train_seq or rl > train_seq:
            pair_trunc += 1
    n = max(len(lines), 1)
    return {
        "chosen_token_len": describe(chosen),
        "rejected_token_len": describe(rejected),
        "pair_truncated_at_train_seq": round(pair_trunc / n, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="token 级长度画像（长上下文截断损毁测量）")
    parser.add_argument("--name", required=True, help="注册表数据集名")
    parser.add_argument("--sample", type=int, default=30000)
    parser.add_argument(
        "--train-seq", type=int, default=None, help="该数据集训练时的截断长度（默认按 kind：380/768/1024）"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    from minimind_reborn.data.registry import file_path, resolve_dataset

    init_tokenizer()
    entry = resolve_dataset(args.name)
    path = file_path(entry)
    train_seq = args.train_seq or {"pretrain": 380, "sft": 768, "dpo": 1024}.get(entry.kind, 1024)

    if entry.kind == "dpo":
        lines = path.open("rb").readlines()  # dpo 仅 1.7 万行，全量普查
    else:
        lines = sample_lines(path, args.sample, args.seed)

    if entry.kind == "pretrain":
        result: dict[str, Any] = profile_pretrain_len(lines, train_seq)
    elif entry.kind == "sft":
        result = profile_sft_len(lines, train_seq)
    else:
        result = profile_dpo_len(lines, train_seq)

    print(
        json.dumps(
            {"dataset": args.name, "train_seq": train_seq, "sampled": len(lines), **result},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

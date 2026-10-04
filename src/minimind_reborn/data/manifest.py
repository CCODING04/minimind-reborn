"""SFT 数据集旁车清单（重训计划 §7.0）。

一次全量扫描（解析 jsonl 取轮数与回复规范化哈希，**不做分词**，14GB 约 10-20 分钟）
产出 npz 旁车文件，为三项修复共用：
- C4 回复去重（keep 列：规范化哈希首现保留）
- C5 哈希分桶 train/val（is_eval 列：规范化行内容稳定哈希，与文件顺序无关）
- D5 数据卡（n_messages 列：多轮占比断言与剔除量记录）

稳定性要点：哈希用 hashlib.md5（跨进程/跨机器稳定），不用 Python hash()（受
PYTHONHASHSEED 盐值影响，同一进程两次运行结果都不同）。以源文件 mtime_ns+size
指纹失效重算；写入走 tmp+rename 原子替换。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 去空白与标点（含全角）；保留 CJK 字符（\w 在 unicode 下涵盖）
_NORMALIZE_RE = re.compile(r"[\W_]+", re.UNICODE)
_BUCKET_MOD = 1000  # 哈希分桶粒度：eval_ratio=0.003 → 命中 [0,3) 进 eval


def normalize_text(text: str) -> str:
    return _NORMALIZE_RE.sub("", text.lower())


def stable_hash(text: str) -> int:
    # signed=True：落在 int64 范围内（npz 存 int64 数组）
    return int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:8], "little", signed=True)


def reply_text(conversations: list[dict]) -> str:
    """去重键：最后一条 assistant 回复（规范化前的原文拼接）。

    C4 对齐 dataeng 的「回复精确重复」口径——复读的训练侧源头是不同问题共享同一回答
    （8.6%），对话级拼接哈希会漏掉这种模式，故取最后一条回复作键。
    """
    replies = [m.get("content", "") for m in conversations if m.get("role") == "assistant"]
    return replies[-1] if replies else ""


@dataclass
class Manifest:
    reply_hash: np.ndarray  # (N,) int64 规范化回复哈希
    keep: np.ndarray  # (N,) bool  C4 去重后保留
    is_eval: np.ndarray  # (N,) bool C5 哈希分桶归属
    n_messages: np.ndarray  # (N,) int32 消息数（多轮 = >2）
    fingerprint: str

    def __len__(self) -> int:
        return len(self.reply_hash)


def _fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


def manifest_path(jsonl_path: str | Path) -> Path:
    p = Path(jsonl_path)
    return p.parent / (p.name + ".manifest.npz")


def is_multi_turn(n_messages: int) -> bool:
    """与 dataeng 画像口径一致：消息数 > 2 视为多轮。"""
    return n_messages > 2


def load_or_build(jsonl_path: str | Path, eval_ratio: float) -> Manifest:
    """读旁车清单；指纹不匹配或缺失时全量扫描重建（原子写）。"""
    path = Path(jsonl_path)
    mpath = manifest_path(path)
    fp = _fingerprint(path)
    if mpath.exists():
        try:
            with np.load(mpath, allow_pickle=False) as z:
                if str(z["fingerprint"]) == fp:
                    return Manifest(
                        z["reply_hash"], z["keep"], z["is_eval"], z["n_messages"], fp
                    )
        except (OSError, KeyError, ValueError):
            pass  # 损坏的旁车文件按缺失处理，重建覆盖

    reply_hashes: list[int] = []
    n_messages: list[int] = []
    is_eval_flags: list[bool] = []
    bucket_take = max(1, round(eval_ratio * 1000))
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            text = line.decode("utf-8", "replace")
            convs = json.loads(text).get("conversations", [])
            reply_hashes.append(stable_hash(normalize_text(reply_text(convs))))
            n_messages.append(len(convs))
            is_eval_flags.append(stable_hash(normalize_text(text)) % _BUCKET_MOD < bucket_take)

    n = len(reply_hashes)
    reply_hash = np.array(reply_hashes, dtype=np.int64)
    n_messages_arr = np.array(n_messages, dtype=np.int32)
    is_eval = np.array(is_eval_flags, dtype=bool)

    # C4：规范化回复哈希首现保留（跨全文件去重，先于分桶——重复行不因分桶翻倍）
    keep = np.ones(n, dtype=bool)
    first_seen: dict[int, int] = {}
    for i, h in enumerate(reply_hashes):
        if h in first_seen:
            keep[i] = False
        else:
            first_seen[h] = i

    tmp = mpath.parent / (mpath.name + ".tmp.npz")  # np.savez 对非 .npz 后缀会自动追加扩展名
    np.savez_compressed(
        tmp,
        reply_hash=reply_hash,
        keep=keep,
        is_eval=is_eval,
        n_messages=n_messages_arr,
        fingerprint=np.array(fp),
    )
    os.replace(tmp, mpath)
    return Manifest(reply_hash, keep, is_eval, n_messages_arr, fp)


def split_balance_report(manifest: Manifest, base_tolerance_pp: float = 2.0) -> dict:
    """train/eval 两桶的多轮占比（仅在 keep 行上统计——与训练可见分布一致）。

    balanced 判定容差 = max(base_tolerance_pp, 3.5σ)，σ 为两桶二项占比差的抽样噪声：
    全量 sft_t2t（eval ≈1.5 万行）σ≈0.33pp，2pp 仍是有效约束；小数据集（如单测的
    百行级）分桶噪声本身就有数个 pp，固定 2pp 会把均匀哈希误判为分布偏移。
    """
    keep_idx = np.flatnonzero(manifest.keep)
    multi = manifest.n_messages[keep_idx] > 2
    is_eval = manifest.is_eval[keep_idx]
    n_train, n_eval = int((~is_eval).sum()), int(is_eval.sum())
    train_share = float(multi[~is_eval].mean()) if n_train else 0.0
    eval_share = float(multi[is_eval].mean()) if n_eval else 0.0
    gap_pp = abs(train_share - eval_share) * 100
    p = float(multi.mean()) if len(multi) else 0.0
    if n_train and n_eval:
        sigma_pp = (p * (1 - p) * (1 / n_train + 1 / n_eval)) ** 0.5 * 100
    else:
        sigma_pp = 0.0
    tolerance_pp = max(base_tolerance_pp, 3.5 * sigma_pp)
    return {
        "kept_total": int(len(keep_idx)),
        "dedup_removed": int(len(manifest) - len(keep_idx)),
        "train_rows": n_train,
        "eval_rows": n_eval,
        "multi_turn_train_share": round(train_share, 4),
        "multi_turn_eval_share": round(eval_share, 4),
        "multi_turn_gap_pp": round(gap_pp, 2),
        "gap_tolerance_pp": round(tolerance_pp, 2),
        "balanced": bool(gap_pp <= tolerance_pp),
    }

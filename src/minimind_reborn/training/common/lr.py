"""LR 纯函数（training §7）：手写 warmup + 余弦，不引入 scheduler 对象状态。"""
from __future__ import annotations

import math


def get_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    *,
    warmup_steps: int = 0,
    min_ratio: float = 0.1,
) -> float:
    """warmup 线性爬升 → 余弦衰减到 base_lr·min_ratio（官方 get_lr 的地板语义保留）。"""
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

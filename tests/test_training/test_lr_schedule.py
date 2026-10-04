"""LR 调度 DDP 语义测试（perf_analysis 案 B 修复的回归锁）。

背景：full_iters 此前未按 world_size 折算，双卡 DDP 下余弦分母是本 rank 实际步数的
2 倍——2 epochs 跑完 LR 停在 75% 进度处（≈lr×0.23）而非 lr×min_ratio 地板。
2026-09-29 上次全量 SFT（sft_full）同受影响；单卡口径恰好正确（mini 时代未暴露）。
"""

from __future__ import annotations

import math

from minimind_reborn.training.common.lr import get_lr
from minimind_reborn.training.common.trainer import per_rank_full_iters

BASE_LR = 1.0e-5
MIN_RATIO = 0.1
FLOOR = BASE_LR * MIN_RATIO


def _end_lr(n_samples: int, batch: int, world: int, epochs: int) -> float:
    """模拟 trainer 的调度序列：每 epoch offset 从 0 到 iters-1，global_micro = epoch×iters+offset。"""
    iters = per_rank_full_iters(n_samples, batch, world)
    last_micro = (epochs - 1) * iters + (iters - 1)
    total = epochs * iters
    return get_lr(last_micro, total, BASE_LR, min_ratio=MIN_RATIO)


def test_world_size_one_unchanged():
    assert per_rank_full_iters(1000, 8, 1) == math.ceil(1000 / 8) == 125


def test_ddp_halves_per_rank_iters():
    # 全量 SFT 口径：4,736,362 样本 / bs 8 → 全量 592,046；双卡每 rank 296,023
    assert per_rank_full_iters(4_736_362, 8, 2) == 296_023


def test_ddp_two_epochs_lands_on_floor():
    """修复后的核心不变式：任意 world 下 2ep 末 LR 收敛到地板（误差 ≤ 半步的余弦斜率）。"""
    for world in (1, 2, 4):
        lr_end = _end_lr(4_736_362, 8, world, epochs=2)
        assert lr_end < FLOOR * 1.001 + 1e-12, f"world={world}: 末值 {lr_end:.3e} 未到地板 {FLOOR:.0e}"


def test_old_bug_signature_locked():
    """回归签名：旧口径（不折算）2ep 末停在 75% 进度 ≈ lr×0.23——若该数字复现即为 bug 回归。"""
    n, batch, epochs = 4_736_362, 8, 2
    buggy_iters = math.ceil(n / batch)  # 旧：全量口径
    last_micro = (epochs - 1) * buggy_iters + (math.ceil(buggy_iters / 2) - 1)  # epoch1 从 full_iters 起步
    total = epochs * buggy_iters
    buggy_lr = get_lr(last_micro, total, BASE_LR, min_ratio=MIN_RATIO)
    assert abs(buggy_lr - 2.32e-6) < 0.05e-6  # 与 2026-09-29 复盘推算一致（非 5.5e-6）


def test_mid_training_lr_matches_fixed_denominator():
    """续训跳变核对：step 152,000 处修复口径 lr≈8.61e-6（旧 9.63e-6），向下跳 11% 可接受。"""
    iters = per_rank_full_iters(4_736_362, 8, 2)
    lr_fixed = get_lr(152_000, 2 * iters, BASE_LR, min_ratio=MIN_RATIO)
    assert abs(lr_fixed - 8.61e-6) < 0.02e-6

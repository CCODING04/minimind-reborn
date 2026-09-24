"""训练基件单元测试：LR 纯函数 / decay 分组 / checkpoint 六件套。"""

from __future__ import annotations

import math

import torch

from minimind_reborn.training.common.lr import get_lr
from minimind_reborn.training.common.optim import configure_optimizers


def test_lr_official_cosine_floor_semantics():
    """官方 get_lr：lr*(0.1 + 0.45*(1+cos(pi*t)))；无 warmup 时曲线完全一致。"""
    lr = 5e-4
    for step, total in [(0, 100), (25, 100), (50, 100), (77, 100), (100, 100)]:
        expected = lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total)))
        assert abs(get_lr(step, total, lr) - expected) < 1e-12


def test_lr_warmup_and_clamp():
    base = 1e-3
    assert abs(get_lr(0, 100, base, warmup_steps=10) - base * 0.1) < 1e-12  # warmup 首步
    assert abs(get_lr(9, 100, base, warmup_steps=10) - base * 1.0) < 1e-12  # warmup 末步
    assert abs(get_lr(100, 100, base) - base * 0.1) < 1e-12  # 余弦地板
    assert abs(get_lr(500, 100, base) - base * 0.1) < 1e-12  # 越界钳制


def test_optimizer_decay_groups():
    """≥2 维 decay；bias 与 1 维（norm）不 decay（training §3 分组不变量）。"""
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8),  # weight(2D)→decay, bias(1D)→no_decay
        torch.nn.LayerNorm(8),  # weight/bias 均 1D → no_decay
    )
    opt = configure_optimizers(model, lr=1e-3, weight_decay=0.01)
    decay_group, no_decay_group = opt.param_groups
    assert len(decay_group["params"]) == 1
    assert len(no_decay_group["params"]) == 3
    assert decay_group["weight_decay"] == 0.01
    assert no_decay_group["weight_decay"] == 0.0


def test_checkpoint_six_pack_roundtrip(tmp_path):
    """六件套（模型/优化器/进度/配置/最佳/RNG）落盘 + 跨卡数 step 换算 + RNG 位精确恢复。"""
    from minimind_reborn.training.common.checkpoint import load_checkpoint, save_checkpoint

    model = torch.nn.Linear(8, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(enabled=False)
    torch.manual_seed(42)
    save_checkpoint(
        tmp_path / "ck.pth",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=1,
        step=7,
        config_snapshot={"a": 1},
        best_val_loss=0.5,
    )
    data = load_checkpoint(tmp_path / "ck.pth", world_size=2)
    assert data["step"] == 3  # 7 * world1 // world2（官方换算语义）
    assert data["epoch"] == 1
    assert data["config"] == {"a": 1}
    assert data["best_val_loss"] == 0.5
    for key in ("model", "optimizer", "scaler", "rng", "world_size"):
        assert key in data
    assert torch.equal(torch.get_rng_state(), data["rng"]["torch"])


def test_extra_states_roundtrip(tmp_path):
    from minimind_reborn.training.common.checkpoint import load_checkpoint, save_checkpoint

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    save_checkpoint(
        tmp_path / "c.pth",
        model=model,
        optimizer=optimizer,
        scaler=None,
        epoch=0,
        step=1,
        config_snapshot={},
        best_val_loss=None,
        extra_states={"scheduler": scheduler},
    )
    data = load_checkpoint(tmp_path / "c.pth")
    assert "scheduler" in data

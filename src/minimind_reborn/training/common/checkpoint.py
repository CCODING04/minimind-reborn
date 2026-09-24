"""checkpoint 六件套（training §4）。

内容：裸模型权重（fp32 主权重，resume 位精确）+ optimizer + scaler +
进度(epoch/step) + 配置快照 + best 指标 + RNG 状态（位精确恢复）。
规则：只有 rank0 写盘（调用方门控）；原子替换防中断留坏文件。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minimind_reborn.loggers import get_logger
from minimind_reborn.utils.io import atomic_save
from minimind_reborn.utils.dist import get_world_size

logger = get_logger("checkpoint")


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def _restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
        except RuntimeError:
            logger.warning("CUDA RNG 状态恢复失败（设备数与保存时不一致），继续但不再位精确")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    step: int,
    config_snapshot: dict,
    best_val_loss: float | None,
    extra_states: dict[str, Any] | None = None,
) -> Path:
    """写入六件套 + RNG（调用方先剥壳：unwrap_model(model)）。

    extra_states：范式自有状态（如 RL 的 scheduler/critic），随包落盘、随包恢复。
    """
    from minimind_reborn.models.weights import unwrap_model

    raw = unwrap_model(model)
    payload = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "world_size": get_world_size(),
        "config": config_snapshot,
        "best_val_loss": best_val_loss,
        "rng": _rng_state(),
    }
    for key, value in (extra_states or {}).items():
        payload[key] = value.state_dict() if hasattr(value, "state_dict") else value
    return atomic_save(payload, path)


def load_checkpoint(path: str | Path, world_size: int | None = None) -> dict[str, Any]:
    """读取续训状态：按卡数变化换算 step（官方语义保留）。"""
    data = torch.load(path, map_location="cpu", weights_only=False)
    saved_ws = data.get("world_size", 1)
    current_ws = world_size if world_size is not None else get_world_size()
    if saved_ws != current_ws:
        data["step"] = data["step"] * saved_ws // current_ws
        logger.info("GPU 数量变化（%d→%d），step 已换算为 %d", saved_ws, current_ws, data["step"])
    if "rng" in data:
        _restore_rng(data["rng"])
        logger.info("RNG 状态已恢复（位精确续训）")
    return data


def checkpoint_path(checkpoint_dir: str | Path, weight: str, hidden_size: int, use_moe: bool) -> Path:
    """官方命名约定：{weight}_{hidden_size}[_moe]_resume.pth。"""
    suffix = "_moe" if use_moe else ""
    return Path(checkpoint_dir) / f"{weight}_{hidden_size}{suffix}_resume.pth"

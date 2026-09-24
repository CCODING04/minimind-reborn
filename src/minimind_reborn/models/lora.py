"""LoRA 适配器（官方 model_lora.py 的重构保留版）。

差异：load 带 map_location="cpu"；其余逻辑（方阵注入 + monkey-patch forward）
保持一致——A 高斯初始化 / B 零初始化保证训练起点等价于基模。
注意：monkey-patch 与 torch.compile 不兼容（官方已知，lora workflow 里强制关闭）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from minimind_reborn.utils.io import atomic_save

logger = logging.getLogger("minimind.lora")


class LoRA(nn.Module):
    """低秩旁路：B(A(x))，A 高斯初始化、B 零初始化。"""

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x))


def apply_lora(model: Any, rank: int = 16) -> None:
    """对方阵 Linear（in==out，即 q/k/v/o_proj 等）注入 LoRA 并接管 forward。"""
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            module.lora = lora  # type: ignore[attr-defined]
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora  # type: ignore[method-assign]


def lora_state_dict(model: Any) -> dict[str, torch.Tensor]:
    """收集全部 LoRA 权重（fp16 CPU），键名带所属模块路径。"""
    raw = model
    state: dict[str, torch.Tensor] = {}
    for name, module in raw.named_modules():
        if hasattr(module, "lora"):
            for k, v in module.lora.state_dict().items():
                state[f"{name}.lora.{k}"] = v.detach().cpu().half()
    return state


def save_lora(model: Any, path: str | Path) -> Path:
    return atomic_save(lora_state_dict(model), path)


def load_lora(model: Any, path: str | Path) -> None:
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            prefix = f"{name}.lora."
            lora_state = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if lora_state:
                module.lora.load_state_dict(lora_state)  # type: ignore[attr-defined]


def merge_lora(model: Any, lora_path: str | Path, save_path: str | Path) -> Path:
    """把 LoRA 合并进基模权重（W' = W + B·A）并保存为基模结构。"""
    load_lora(model, lora_path)
    raw = model
    merged: dict[str, torch.Tensor] = {}
    for name, module in raw.named_modules():
        lora = getattr(module, "lora", None)
        if isinstance(module, nn.Linear) and lora is not None:
            delta = (lora.B.weight.data @ lora.A.weight.data).cpu().half()
            merged[f"{name}.weight"] = (module.weight.data.clone().cpu().half() + delta).half()
    base = {k: v.cpu().half() for k, v in raw.state_dict().items() if ".lora." not in k}
    base.update(merged)
    return atomic_save(base, save_path)

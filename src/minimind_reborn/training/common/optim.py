"""优化器构造（training §3：decay 分组放构造处，不散在训练循环）。"""
from __future__ import annotations

import inspect

import torch


def configure_optimizers(model: torch.nn.Module, lr: float, weight_decay: float, params=None) -> torch.optim.Optimizer:
    """AdamW + 权重 decay 分组：≥2 维张量 decay，bias 与 1 维参数（norm）不 decay。

    params=None 时对全部可训练参数分组（LoRA 场景传入自己的参数集）；
    fused 实现经签名探测可用才开（能力探测优于版本判断）。
    """
    trainable = [p for p in (params if params is not None else model.parameters()) if p.requires_grad]
    decay = [p for p in trainable if p.dim() >= 2]
    no_decay = [p for p in trainable if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra = {"fused": True} if (fused_available and torch.cuda.is_available()) else {}
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), **extra)

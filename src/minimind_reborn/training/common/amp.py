"""混合精度上下文集中构造（training §2）。

不变量：
- GradScaler 恒创建、用 enabled=(dtype=='fp16') 控制——bf16 没有 loss scale 机制，
  挂 scaler 是无效开销，fp16 不挂则 loss 上溢 NaN（anti-patterns §训练 前两行）；
- 精度上下文在这里构造，训练循环里只使用不判断（CPU/禁用 AMP 走 nullcontext 分支）；
- 设备不支持 bf16 时回退 fp16 并警告一次（降级必须留痕）。
"""
from __future__ import annotations

import logging
from contextlib import nullcontext

import torch

logger = logging.getLogger("minimind.amp")


def resolve_dtype(name: str) -> torch.dtype:
    """配置字符串 → torch dtype；设备不支持 bf16 时回退 fp16 并留痕。"""
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            logger.warning("当前 GPU 不支持 bf16，AMP 回退 fp16（降级留痕）")
            return torch.float16
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"未知精度 {name!r}（守门人应已拦截，说明绕过了 loader）")


def autocast_context(dtype: torch.dtype, device: str):
    """autocast 上下文：CPU 或 fp32 时 nullcontext，否则 cuda autocast。"""
    if device.startswith("cuda") and dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def build_scaler(dtype: torch.dtype) -> torch.amp.GradScaler:
    """scaler 恒创建，enabled 只认 fp16——scaler 的 step/update 调用次数与 optimizer 对齐。"""
    return torch.amp.GradScaler(enabled=(dtype == torch.float16))

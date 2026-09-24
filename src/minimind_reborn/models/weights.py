"""权重保存/加载（training §4：六件套 + resume/finetune 双语义）。

不变量：
- torch.load 永远 map_location="cpu"（GPU 序列化文件在无卡机器上直接炸）；
- 保存裸模型（剥 DDP/_orig_mod 壳），不存分布式包装；
- resume 语义严格恢复（结构超参以 checkpoint 为准）；
- finetune 语义非严格加载，但缺失/形状不匹配的 key 逐条 warning——
  静默吞键是最危险的"成功"。
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from minimind_reborn.utils.io import atomic_save

logger = logging.getLogger("minimind.weights")


def to_device[M: torch.nn.Module](model: M, target: str) -> M:
    """nn.Module 口径的设备搬运：规避 transformers 5.x PreTrainedModel.to 的 stub 回归。"""
    return model.to(target)


def to_dtype[M: torch.nn.Module](model: M, dtype: torch.dtype) -> M:
    """nn.Module 口径的精度搬运（同上）。"""
    return model.to(dtype=dtype)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """剥掉 DDP / torch.compile 壳，返回裸模型（保存与推理路径统一入口）。"""
    from torch.nn.parallel import DistributedDataParallel

    raw: torch.nn.Module = model
    if isinstance(raw, DistributedDataParallel):
        raw = raw.module
    return getattr(raw, "_orig_mod", raw)


def save_inference_weights(model: torch.nn.Module, path: str | Path, dtype: torch.dtype = torch.float16) -> Path:
    """保存推理权重（fp16 CPU，原子写）。调用方负责 rank0 门控。"""
    raw = unwrap_model(model)
    state_dict = {k: v.detach().to(dtype=dtype, device="cpu") for k, v in raw.state_dict().items()}
    return atomic_save(state_dict, path)


def load_inference_weights(model: torch.nn.Module, path: str | Path, *, strict: bool = True) -> None:
    """推理加载：默认严格；权重先到 CPU 再由调用方 .to(device/dtype)（精度回退链在 engine）。"""
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"权重加载失败（严格模式）：{len(missing)} 个缺失 key / {len(unexpected)} 个多余 key。"
            f"模型结构可能与权重不匹配；微调场景请用 load_finetune_weights（非严格 + 逐 key warning）"
        )
    if missing or unexpected:
        _warn_key_diff(missing, unexpected, [])


def load_finetune_weights(model: torch.nn.Module, path: str | Path) -> None:
    """微调加载：非严格 + 缺失/形状不匹配逐 key warning（training §4 反"静默吞键"）。"""
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model_state = model.state_dict()
    shape_mismatch = [k for k, v in state_dict.items() if k in model_state and model_state[k].shape != v.shape]
    filtered = {k: v for k, v in state_dict.items() if k not in shape_mismatch}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    _warn_key_diff(missing, unexpected, shape_mismatch)


def _warn_key_diff(missing: list[str], unexpected: list[str], shape_mismatch: list[str]) -> None:
    if missing:
        logger.warning("权重加载缺失 %d 个 key（保持初始化值）：%s", len(missing), missing[:8])
    if unexpected:
        logger.warning("权重加载多余 %d 个 key（已忽略）：%s", len(unexpected), unexpected[:8])
    if shape_mismatch:
        logger.warning("权重加载形状不匹配 %d 个 key（已忽略）：%s", len(shape_mismatch), shape_mismatch[:8])


def resolve_weight_path(name_or_path: str, out_root: str | Path, hidden_size: int, use_moe: bool) -> Path:
    """权重名 → 路径。官方命名约定 {weight}_{hidden_size}[_moe].pth；绝对路径原样返回。"""
    path = Path(name_or_path)
    if path.is_file():
        return path
    suffix = "_moe" if use_moe else ""
    return Path(out_root) / f"{name_or_path}_{hidden_size}{suffix}.pth"

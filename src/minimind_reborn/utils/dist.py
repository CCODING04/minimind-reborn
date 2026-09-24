"""分布式容错封装（training §6）。

不变量：业务代码不写 `if distributed:` 分支——get_rank()/get_world_size()
在 dist 不可用/未初始化时返回 0/1，单机与多机代码同构；
一切 IO 由 is_main_process() 门控。
"""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def init_distributed(timeout_minutes: int = 30) -> int:
    """按环境变量探测 DDP（torchrun 注入 RANK/WORLD_SIZE），单卡自动回退。

    返回 local_rank（非分布式时为 0）。CUDA 不可用时显式报错——
    torchrun + NCCL 必须有 GPU，静默回退 CPU 只会把错误推迟到更难查的地方。
    """
    if os.environ.get("RANK", "-1") == "-1":
        return 0
    if not dist.is_available() or not torch.cuda.is_available():
        raise RuntimeError("检测到 RANK 环境变量（torchrun 启动）但 CUDA 不可用：分布式后端 NCCL 需要 GPU")
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=timeout_minutes))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_initialized() -> bool:
    """torch.distributed 已初始化（透传容错：未安装 dist 时恒 False）。"""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def all_reduce_mean(value: float, device: str = "cpu") -> float:
    """跨 rank 求平均（评估指标聚合）。非分布式时原样返回。"""
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return float(t.item())


# 透传原生分布式原语：业务代码经本模块访问，避免散落 import torch.distributed
all_reduce = dist.all_reduce
ReduceOp = dist.ReduceOp


def broadcast_object(obj, src: int = 0):
    """rank0 的对象广播到全部 rank（如时间戳派生的 run_dir——各 rank 各自取时间会得到不同目录）。"""
    if not is_initialized():
        return obj
    container = [obj]
    dist.broadcast_object_list(container, src=src)
    return container[0]

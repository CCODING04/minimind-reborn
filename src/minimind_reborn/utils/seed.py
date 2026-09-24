"""全链路随机种子（training §1：可复现性先于一切）。

不变量：相同 seed + 相同代码 + 相同硬件 → 逐位相同的训练前若干步 loss。
两个易漏点都在这里处理：
- dataloader worker 种子：fork 后叠加 rank/worker 无关熵，防增强样本跨 worker 重复
  （原版 num_workers=8 + SFT 运行时随机增强正踩此坑）；
- 分布式按 rank 偏移：数据顺序/dropout 各 rank 错开，模型初始化保持一致。
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """设置 python/numpy/torch 全部随机源。

    deterministic=True 时开启 cudnn 确定性——有可观测的性能代价，
    仅在复现实验位对位时打开（默认关闭，打开时警告一次）。
    """
    if deterministic:
        logger.warning("cudnn.deterministic 已开启：可复现但训练会变慢（training §1 减速警告）")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn：每个 worker 的种子叠加进程无关熵。

    torch 的默认 worker seed 已含 base_seed（由主进程生成），
    这里再叠加 numpy/python 两个源，保证多 worker 的运行时增强互不重复。
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def worker_init() -> dict:
    """DataLoader 构造参数：所有入口统一使用，不再各自手写。"""
    return {"worker_init_fn": seed_worker}


def loader_kwargs(num_workers: int) -> dict:
    """DataLoader 通用 kwargs（pin_memory + worker 种子），集中一处。"""
    kwargs: dict = {"num_workers": num_workers, "pin_memory": True}
    if num_workers > 0:
        kwargs["worker_init_fn"] = seed_worker
    return kwargs

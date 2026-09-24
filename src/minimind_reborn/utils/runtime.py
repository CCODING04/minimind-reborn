"""运行时环境准备入口（training §8：ulimit、线程数在训练入口一次设置）。

参数均有"实测依据 + 允许覆盖"注释；业务代码不直接调用本模块（入口统一调用）。
"""

from __future__ import annotations

import logging
import resource

import torch

from minimind_reborn import envs

logger = logging.getLogger("minimind.runtime")


def configure_runtime() -> None:
    """训练入口一次调用：线程数 + 文件句柄上限。

    - torch 线程数默认 8：实测 8 worker 时数据管道 12× 富余于训练吞吐
      （worker 扫描见 docs/pretrain_readiness.md §2），主进程留足 CPU 给
      DataLoader 与系统；env MINIMIND_REBORN_TORCH_THREADS 可覆盖。
    - RLIMIT_NOFILE 抬到 min(hard, 65536)：DataLoader 多 worker + 大量分词句柄
      在默认 1024 软限制下会报 Too many open files。
    """
    threads = envs.torch_threads()
    torch.set_num_threads(threads)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(hard, 65536)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            soft = target
        logger.info("运行时准备：torch 线程=%d, RLIMIT_NOFILE=%d（硬上限 %d）", threads, soft, hard)
    except (OSError, ValueError) as e:
        logger.warning("RLIMIT_NOFILE 调整失败（不影响训练，降级留痕）：%s", e)

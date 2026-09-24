"""环境变量单一来源（python-style §3：全部 os.environ 读取集中在此）。

惰性读取 + 进程内缓存 + 类型化；业务代码不允许直接摸 os.environ。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_CACHE: dict[str, Any] = {}


def _get_str(key: str, default: str) -> str:
    if key not in _CACHE:
        _CACHE[key] = os.environ.get(key, default)
    return str(_CACHE[key])  # type: ignore[return-value]


def _get_int(key: str, default: int) -> int:
    if key not in _CACHE:
        raw = os.environ.get(key)
        _CACHE[key] = int(raw) if raw not in (None, "") else default
    return int(_CACHE[key])  # type: ignore[return-value]


def data_root() -> Path:
    """数据集根目录（jsonl 所在地）。默认指向工作区已有的 minimind 数据目录。"""
    return Path(_get_str("MINIMIND_REBORN_DATA_ROOT", "~/Code/WorkSpace/minimind-reproduce/dataset")).expanduser()


def out_root() -> Path:
    """训练产物（推理权重 .pth）输出目录。"""
    return Path(_get_str("MINIMIND_REBORN_OUT_ROOT", "out"))


def checkpoint_root() -> Path:
    """续训 checkpoint（resume 六件套）输出目录。"""
    return Path(_get_str("MINIMIND_REBORN_CHECKPOINT_ROOT", "checkpoints"))


def run_root() -> Path:
    """一次运行的可观测四件套（train.log/metrics.jsonl/config.json/env.json）目录。"""
    return Path(_get_str("MINIMIND_REBORN_RUN_ROOT", "runs"))


def tokenizer_path() -> Path:
    """官方 tokenizer 目录（随包携带，与官方权重/数据兼容）。"""
    default = Path(__file__).resolve().parent / "assets" / "tokenizer"
    return Path(_get_str("MINIMIND_REBORN_TOKENIZER_PATH", str(default)))


def metrics_disabled() -> bool:
    """测试/CI 强制禁用云端上报（logging-metrics §6 测试纪律）。"""
    return _get_int("MINIMIND_REBORN_METRICS_DISABLED", 0) == 1


def log_all_ranks() -> bool:
    """调试分布式时放开非 rank0 的控制台日志（默认只 rank0，防多卡交错刷屏）。"""
    return _get_int("MINIMIND_REBORN_LOG_ALL_RANKS", 0) == 1


def torch_threads() -> int:
    """主进程 torch intraop 线程数（worker 扫描实测：4 worker 即饱和，8 线程富余，见 docs/pretrain_readiness.md §2）。"""
    return _get_int("MINIMIND_REBORN_TORCH_THREADS", 8)


def seed_offset() -> int:
    """全局种子偏移（同一配方跑不同 seed 的实验组）。"""
    return _get_int("MINIMIND_REBORN_SEED_OFFSET", 0)


# 库代码在 import 期唯一允许设置的环境变量：关闭 tokenizers 的 fork 并行告警
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

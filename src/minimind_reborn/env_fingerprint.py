"""环境指纹（logging-metrics §4 可追溯四件套之一）。

训练入口自动生成 env.json：git commit + dirty 标记、python/torch/cuda 版本、
GPU 型号与数量、主机名、seed。事后只凭输出目录能回答"什么环境"。
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any

from minimind_reborn.loggers import get_logger

logger = get_logger("env")


def _git_info(repo_dir: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": False}


def collect(seed: int | None = None) -> dict[str, Any]:
    """采集环境指纹。GPU 信息缺失时记录占位而非失败（无卡机器也要能留档）。"""
    import torch
    import transformers

    gpu_names: list[str] = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    fingerprint: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": gpu_names or "none",
        "gpu_count": len(gpu_names),
        "seed": seed,
    }
    fingerprint.update(_git_info(Path(__file__).resolve().parent))
    return fingerprint


def dump(run_dir: str | Path, seed: int | None = None) -> Path:
    """写入 run_dir/env.json 并往日志打一行摘要（四件套纪律）。"""
    fp = collect(seed=seed)
    from minimind_reborn.utils.io import atomic_write_json

    path = atomic_write_json(fp, Path(run_dir) / "env.json")
    logger.info(
        "环境指纹：%s", json.dumps({k: fp[k] for k in ("torch", "cuda_runtime", "gpu", "seed")}, ensure_ascii=False)
    )
    return path

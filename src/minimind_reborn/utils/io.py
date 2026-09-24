"""磁盘 IO 原子性（training §4：写盘用临时文件 + rename，防中断留坏文件）。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_save(obj: Any, path: str | Path) -> Path:
    """torch.save 的原子版本：先写 .tmp 再 os.replace（同目录保证原子性）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    import torch

    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def atomic_write_json(data: Any, path: str | Path) -> Path:
    """json 序列化的原子版本（配置快照/环境指纹/指标摘要）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

"""数据集声明式注册表（project-structure §4：数据描述的资源走配置文件）。

未注册名报错时指明本 json 路径；kind 决定 Dataset 类的派发。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from minimind_reborn import envs
from minimind_reborn.loggers import get_logger

logger = get_logger("data.registry")

_REGISTRY_FILE = "datasets.json"


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    file: str
    kind: str  # pretrain | sft | dpo | rlaif | agent
    description: str


def _load_registry() -> dict[str, DatasetEntry]:
    raw = json.loads(resources.files("minimind_reborn.data").joinpath(_REGISTRY_FILE).read_text(encoding="utf-8"))
    return {
        name: DatasetEntry(name=name, file=item["file"], kind=item["kind"], description=item.get("description", ""))
        for name, item in raw.items()
    }


_REGISTRY = _load_registry()


def resolve_dataset(name: str, data_root: Path | None = None) -> DatasetEntry:
    """注册表名 → 条目；本地文件缺失时报三要素错误。"""
    if name not in _REGISTRY:
        raise ValueError(
            f"数据集 '{name}' 未注册。已注册：{sorted(_REGISTRY)}。"
            f"新增数据集请在 {Path(_REGISTRY_FILE)} 中声明（名称/文件名/kind），不要散落硬编码路径"
        )
    entry = _REGISTRY[name]
    path = (data_root or envs.data_root()) / entry.file
    if not path.exists():
        raise FileNotFoundError(
            f"数据集文件不存在：{path}（注册名 '{name}'）。"
            f"请检查 MINIMIND_REBORN_DATA_ROOT 或运行 tools/download_data.py --name {name} 从 modelscope 下载"
        )
    return entry


def file_path(entry: DatasetEntry, data_root: Path | None = None) -> Path:
    return (data_root or envs.data_root()) / entry.file

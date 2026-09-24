"""数据集下载（modelscope）：本地缺失的注册表数据集从这里补齐。

用法：uv run python tools/download_data.py --name rlaif_mini
"""

from __future__ import annotations

import argparse
from pathlib import Path

from minimind_reborn import envs
from minimind_reborn.data.registry import _REGISTRY

# 官方 minimind 数据集仓库（modelscope datasets，namespace 为 gongjy）
MODELSCOPE_REPO = "gongjy/minimind_dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description="从 modelscope 下载注册表数据集")
    parser.add_argument("--name", required=True, choices=sorted(_REGISTRY), help="注册表数据集名")
    args = parser.parse_args()

    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise SystemExit(f"缺少下载依赖（{e}）。请安装：uv sync --extra download") from None

    entry = _REGISTRY[args.name]
    target = envs.data_root() / entry.file
    if target.exists():
        print(f"已存在，跳过：{target}")
        return
    print(f"从 modelscope 下载 {MODELSCOPE_REPO}/{entry.file} …")
    repo_dir = Path(snapshot_download(MODELSCOPE_REPO, repo_type="dataset", allow_patterns=[entry.file]))
    source = repo_dir / entry.file
    if not source.exists():
        raise SystemExit(f"下载完成但未找到 {entry.file}（仓库布局可能已变化），请检查 {repo_dir}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    print(f"完成：{target}")


if __name__ == "__main__":
    main()

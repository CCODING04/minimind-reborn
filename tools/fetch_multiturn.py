"""外部多轮对话数据下载（重训计划档二 D1 对策）。

主源：BelleGroup/multiturn_chat_0.8M（约 80 万条中文多轮对话，Human:/Assistant: 标记格式），
经 hf-mirror.com 拉取（hf.co 不可直连的网络环境）。落盘到数据根目录 raw/ 下，
清洗转换见 tools/merge_multiturn.py。

用法：uv run --extra download python tools/fetch_multiturn.py [--force]
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from urllib.parse import quote

from minimind_reborn import envs

MIRROR = "https://hf-mirror.com"
REPO = "BelleGroup/multiturn_chat_0.8M"
FILENAME = "multiturn_chat_0.8M.json"
CHUNK = 1 << 20  # 1MB


def target_path() -> Path:
    return envs.data_root() / "raw" / FILENAME


def fetch(dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"已存在，跳过（--force 重下）：{dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MIRROR}/datasets/{REPO}/resolve/main/{quote(FILENAME)}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    # 断点续传：已有 .part 则从其大小起 Range 续传
    start = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("ab" if start else "wb") as f:
        total = int(resp.headers.get("Content-Length", 0)) + start
        done = start
        mode = "a" if start else "w"
        print(f"下载 {url}\n{'续传' if start else '开始'}：{total/1e6:.0f}MB -> {tmp}")
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done % (50 * CHUNK) < CHUNK:
                print(f"  {mode} {done/1e6:.0f}/{total/1e6:.0f}MB ({done/total:.0%})", flush=True)
    size = tmp.stat().st_size
    if size < 100_000_000:  # 完整文件 ~990MB，防半截落盘
        raise RuntimeError(f"下载不完整：{size} 字节")
    tmp.rename(dest)
    print(f"完成：{dest} ({size/1e6:.0f}MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载外部多轮对话数据（Belle multiturn）")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = parser.parse_args()
    fetch(target_path(), force=args.force)


if __name__ == "__main__":
    main()

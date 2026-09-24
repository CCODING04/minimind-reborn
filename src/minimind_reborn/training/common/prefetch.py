"""CUDA 侧流预取器（training §8：H2D 拷贝与当前步计算重叠）。

YOLOX DataPrefetcher 同款模式：维护"下一批"在独立 CUDA stream 上做 non_blocking
拷贝，训练循环取当前批时先 wait_stream 对齐——数据传输与上一步的反向计算重叠。
- 仅张量字段搬运，非张量字段（RL 的 messages/tools 等）原样透传；
- CPU 设备自动旁路（直接透传 loader），符合"GPU 之外无预取语义"。
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


class CUDAPrefetcher:
    def __init__(self, loader: DataLoader, device: str):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device) if device.startswith("cuda") else None
        self._iter = iter(loader)
        self._next: dict | None = None
        if self.stream is not None:
            self._preload()

    def _preload(self) -> None:
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next = None
            return
        with torch.cuda.stream(self.stream):
            self._next = {
                k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

    def __iter__(self) -> CUDAPrefetcher:
        return self

    def __next__(self) -> dict:
        if self.stream is None:  # CPU 旁路
            return next(self._iter)
        torch.cuda.current_stream().wait_stream(self.stream)  # 等侧流拷贝完成
        if self._next is None:
            raise StopIteration
        batch, self._next = self._next, None
        self._preload()  # 预载再下一批（与本步计算重叠）
        return batch

    def __len__(self) -> int:
        return len(self.loader)

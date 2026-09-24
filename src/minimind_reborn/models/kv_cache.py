"""KV cache：一次性预分配 + 原地切片（llm-inference §2 不变量）。

对比官方实现（每层 (k, v) tuple、每步 torch.cat 重新分配整段历史）：
- 容量按 batch × max_seq_len × kv_heads × head_dim 一次分配，推理全程零再分配；
- 写入是 `[:, pos:pos+len]` 原地切片，读取是前缀视图（零拷贝）；
- 是普通 Python 属性而非 register_buffer——不进 state_dict、不被 save 带走；
- dtype 在构造时对齐当前推理精度一次，之后不再搬运。
"""

from __future__ import annotations

import torch


class KVCache:
    """所有层共享一个张量组，layer_idx 索引第 0 维。"""

    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ):
        shape = (num_layers, max_batch_size, max_seq_len, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.cur_len = 0  # 已缓存的真实长度（不含预分配的空位）
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start: int) -> None:
        """写入一层的新 k/v。k, v: (batch, new_len, kv_heads, head_dim)。"""
        new_len = k.shape[1]
        if start + new_len > self.max_seq_len:
            raise ValueError(
                f"KV cache 溢出：写入位置 {start}+{new_len} 超过预分配容量 {self.max_seq_len}。"
                f"请增大 GenerateConfig.max_new_tokens 对应的容量（generator 按 prompt+max_new_tokens 预分配）"
            )
        # 原地切片写入；dtype 不一致时 copy_ 自动转换（惰性对齐一次）
        self.k[layer_idx, :, start : start + new_len].copy_(k)
        self.v[layer_idx, :, start : start + new_len].copy_(v)

    def read(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """读取一层的全部有效历史（前缀视图，零拷贝）。"""
        return self.k[layer_idx, :, : self.cur_len], self.v[layer_idx, :, : self.cur_len]

    def advance(self, length: int) -> None:
        """一个 forward 步写入完成后统一推进游标（各层写同一 start，避免逐层竞争）。"""
        self.cur_len += length

    def reset(self) -> None:
        self.cur_len = 0

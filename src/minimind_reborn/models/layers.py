"""组合子模块：RMSNorm / RoPE / GQA 工具。

数值纪律（training §2）：norm、rotary 等半精度下易失稳的算子显式升 fp32
计算、算完 `type_as` 回落——写法固定在此处，不依赖 autocast 内部行为。
"""

from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm：fp32 归一化后回落输入精度。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.weight * self._norm(x.float())).type_as(x)


def precompute_freqs_cis(
    dim: int,
    end: int = 32768,
    rope_base: float = 1e6,
    rope_scaling: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 的 cos/sin 表，形状 (end, dim)，sin/cos 各重复一半以对齐 rotate_half。

    rope_scaling 非空时按 YaRN 插值：f'(i) = f(i)·((1-γ) + γ/s)，
    γ 是 [low, high] 维度间的线性 ramp（维度分组由 beta_fast/beta_slow 推出）。
    """
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)
        if end / orig_max > 1.0:
            # 由 beta 反解 ramp 的维度边界：inv_dim(b) = dim·ln(orig_max/(b·2π)) / (2·ln(base))
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))  # noqa: E731
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end)
    angles = torch.outer(t, freqs).float()  # (end, dim//2)
    freqs_cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 q/k 施加 RoPE。

    q, k: (b, seq, heads, head_dim)；cos, sin: (seq, head_dim) → 在 head 维前 unsqueeze。
    """

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        # (b, seq, heads, head_dim) 前后半交换符号拼接
        return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA：把 kv 头复制 n_rep 份对齐 q 头数。

    x: (b, seq, kv_heads, head_dim) → (b, seq, kv_heads·n_rep, head_dim)；n_rep=1 原样返回。
    """
    batch, seq_len, kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch, seq_len, kv_heads, n_rep, head_dim)
        .reshape(batch, seq_len, kv_heads * n_rep, head_dim)
    )

"""GQA + QK-Norm 注意力：三条执行路径，一条正确性论证。

路径选择（按序判定）：
1. SDPA is_causal——无 cache 的训练/整段编码（attention_mask 全 1 或 None）；
2. SDPA 融合 prefill/decode——带 cache 的步进路径，用显式 additive mask 表达
   "causal 只作用于当前步新位置块 + 历史/新块列的 padding 抑制"。融合 kernel 不物化
   (heads, seq, total) 的 fp32 打分矩阵，显存与速度都优于路径 3（官方同场景用每步
   torch.cat cache + SDPA prefill，数值路径因此与我们原 manual 实现存在 fp16 差异）；
3. manual 显式打分——保留原实现：flash_attn=False / 无 SDPA 时走到。fp32 打分 +
   softmax 回落是数值基线，也是理解路径 2 的参考实现（causal mask 的 off-by-one
   注释保留在此）。

fp16 双路径贪心分叉属预期浮点行为：数值验证以 fp32 为准（fp32 下三路径逐位一致）。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.kv_cache import KVCache
from minimind_reborn.models.layers import RMSNorm, apply_rotary_pos_emb, repeat_kv


def sdpa_available() -> bool:
    """能力探测一处实现全项目复用（不散落 hasattr）。"""
    return hasattr(F, "scaled_dot_product_attention")


class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_kv_heads = config.num_key_value_heads
        self.num_heads = config.num_attention_heads
        self.n_rep = self.num_heads // self.num_kv_heads  # 每个 kv 头服务几组 q 头
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        # QK-Norm：在 head_dim 上做 RMSNorm，稳定小模型注意力熵
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = sdpa_available() and config.flash_attn

    def forward(
        self,
        x: torch.Tensor,  # (b, seq, hidden)
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # (cos, sin)，各 (seq, head_dim)
        cache: KVCache | None = None,
        layer_idx: int = 0,
        cache_start: int = 0,  # 本步写入 cache 的起始位置
        attention_mask: torch.Tensor | None = None,  # (b, total_len)，1=有效 0=padding
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if cache is not None:
            # 原地写入本步新 k/v，再以零拷贝前缀视图读回全部历史
            cache.write(layer_idx, xk, xv, cache_start)
            total_len = cache_start + seq_len
            xk_full, xv_full = cache.k[layer_idx, :, :total_len], cache.v[layer_idx, :, :total_len]
        else:
            xk_full, xv_full = xk, xv

        xq = xq.transpose(1, 2)  # (b, heads, seq, hd)
        xk_full = repeat_kv(xk_full, self.n_rep).transpose(1, 2)  # (b, heads, total, hd)
        xv_full = repeat_kv(xv_full, self.n_rep).transpose(1, 2)

        use_sdpa = (
            self.flash
            and cache is None  # 路径 1：训练/整段编码
            and (attention_mask is None or bool(torch.all(attention_mask == 1)))
        )
        if use_sdpa:
            output = F.scaled_dot_product_attention(
                xq, xk_full, xv_full, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal
            )
        elif self.flash:
            # 路径 2：SDPA 融合 prefill/decode（带 cache）。additive mask 一次构造：
            #   - causal 部分只落在"当前步新位置块"的最后 seq_len 列——历史列全部可见；
            #     新块内第 i 行允许看到块内前 i+1 个位置（与路径 3 的 [-seq_len:] 切片同一论证）
            #   - padding 列用 masked_fill 覆盖为 finfo.min（覆盖而非累加，避免 fp16 下限相加下溢）
            total_len = cache_start + seq_len
            neg_min = torch.finfo(xq.dtype).min
            mask = torch.zeros(bsz, 1, seq_len, total_len, device=xq.device, dtype=xq.dtype)
            if seq_len > 1:
                causal = torch.zeros(seq_len, total_len, device=xq.device, dtype=xq.dtype)
                causal[:, cache_start:] = torch.full(
                    (seq_len, seq_len), neg_min, device=xq.device, dtype=xq.dtype
                ).triu(1)
                mask = mask + causal
            if attention_mask is not None:
                mask = mask.masked_fill(attention_mask[:, None, None, :total_len] == 0, neg_min)
            output = F.scaled_dot_product_attention(
                xq, xk_full, xv_full, dropout_p=self.dropout if self.training else 0.0, attn_mask=mask
            )
        else:
            # 路径 3：manual 显式打分（保留原实现——数值基线与教学参考）。
            # 数值纪律：打分升 fp32，softmax 后回落（与官方一致）
            scores = (xq @ xk_full.transpose(-2, -1)).float() / math.sqrt(self.head_dim)  # (b, heads, seq, total)
            if self.is_causal:
                # causal mask 只作用于"当前步的新位置块"——历史列全部可见。
                # 切片 [:, :, :, -seq_len:] 保证步进（seq=1）与 prefill（seq>1）共用同一正确性论证：
                # 新块内第 i 行允许看到块内前 i+1 个位置。
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(
                    1
                )
            if attention_mask is not None:
                # padding 位置加性 -1e9（0 mask → 抑制）；广播到 head 维
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).to(scores.dtype)) * -1e9
            output = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq)) @ xv_full

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.resid_dropout(self.o_proj(output))

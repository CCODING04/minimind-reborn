"""前馈子模块：SwiGLU FFN 与 MoE。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from minimind_reborn.models.config import MiniMindConfig


class FeedForward(nn.Module):
    """SwiGLU：down(act(gate(x)) · up(x))，全部无 bias（与官方/LLaMA 一致）。"""

    def __init__(self, config: MiniMindConfig, intermediate_size: int | None = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MOEFeedForward(nn.Module):
    """细粒度 MoE：softmax 路由 + top-k 专家 + 负载均衡 aux loss。

    训练侧两个技巧（保留官方实现，加注来源语义）：
    - top-1 时直通估计器：前向恒乘 1.0，梯度经 (top1 - top1.detach() + 1.0) 回传；
    - 未被路由的专家经 `y[0,0] += 0·Σp` 挂在计算图上，DDP 不报未使用参数。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)]
        )
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (b, seq, hidden) → (b, seq, hidden)
        batch, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)  # (b·seq, num_experts) 路由分数
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1:
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                # k=1 归一化是恒等，改用直通估计器把权重钉在 1.0 而梯度仍走 softmax
                top1 = torch.topk(F.softmax(self.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False)[0]
                topk_weight = top1 - top1.detach() + 1.0
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = topk_idx == i  # (b·seq, k)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            # 负载均衡：路由频率 load 与平均分数 importance 的点积，越小越均衡
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch, seq_len, hidden_dim)

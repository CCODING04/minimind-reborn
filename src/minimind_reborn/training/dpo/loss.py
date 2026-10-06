"""DPO 损失（与官方 train_dpo.py 的实现语义一致，纯函数可单测）。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def logits_to_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits (b, seq, vocab), labels (b, seq) → 每 token 的 log prob (b, seq)。"""
    log_probs = F.log_softmax(logits, dim=2)
    return torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)


def dpo_loss(
    ref_log_probs: torch.Tensor,
    policy_log_probs: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """sigmoid-DPO：-log σ(β·[(π_c-π_r) - (ref_c-ref_r)])，mask 限定回答区间求和。

    输入 (b, seq)，batch 前半 chosen 后半 rejected（要求偶数——奇数批经
    `half = b // 2` 切分后形状不匹配，此前以难懂的张量广播错误抛出）。
    """
    if ref_log_probs.shape[0] % 2 != 0:
        raise ValueError(
            f"DPO 拼批要求偶数 batch（前半 chosen、后半 rejected 各半），得到 {ref_log_probs.shape[0]}"
        )
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)
    half = ref_log_probs.shape[0] // 2
    pi_logratios = policy_log_probs[:half] - policy_log_probs[half:]
    ref_logratios = ref_log_probs[:half] - ref_log_probs[half:]
    return -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

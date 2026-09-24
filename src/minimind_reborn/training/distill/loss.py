"""蒸馏损失：温度软化的 KL（官方 distillation_loss 保留）。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """softmax(T) 教师分布 → KL(student‖teacher) × T²，batchmean 归约。"""
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return (temperature**2) * F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

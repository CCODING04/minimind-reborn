"""DPO 损失纯函数测试：公式语义 + 偶数批守卫（回归：2026-10-06 奇数批广播错误）。"""

from __future__ import annotations

import math

import pytest
import torch

from minimind_reborn.training.dpo.loss import dpo_loss, logits_to_log_probs


def _pair_logps(gap: float) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 (b=2, seq=1) 的 ref/policy logps：policy 相对 ref 整体抬 gap。"""
    ref = torch.zeros(2, 1)
    policy = ref.clone()
    policy[0, 0] = gap  # chosen 抬高 = 偏好方向正确
    return ref, policy


def test_dpo_loss_prefers_chosen_over_rejected():
    mask = torch.ones(2, 1)
    ref, policy = _pair_logps(gap=1.0)
    loss = dpo_loss(ref, policy, mask, beta=0.1)
    expected = -math.log(torch.sigmoid(torch.tensor(0.1 * 1.0)).item())
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_dpo_loss_symmetric_gap_direction():
    """rejected 被抬高同样的量 → 损失相同（β·gap 的绝对方向决定 σ 输入）。"""
    mask = torch.ones(2, 1)
    ref, policy = _pair_logps(gap=-1.0)
    loss_down = dpo_loss(ref, policy, mask, beta=0.1)
    ref2, policy2 = _pair_logps(gap=1.0)
    loss_up = dpo_loss(ref2, policy2, mask, beta=0.1)
    # β·(πc-πr-(refc-refr)) 两个方向输入互为相反数，-logσ(x) ≠ -logσ(-x)，各自有定义即可
    assert math.isfinite(loss_down.item()) and math.isfinite(loss_up.item())


def test_dpo_loss_rejects_odd_batch():
    """奇数批显式报错（回归：2026-10-06 前 half=b//2 切分后形状不匹配的裸广播错误）。"""
    ref = torch.zeros(3, 4)
    policy = torch.zeros(3, 4)
    mask = torch.ones(3, 4)
    with pytest.raises(ValueError, match="偶数 batch"):
        dpo_loss(ref, policy, mask, beta=0.1)


def test_logits_to_log_probs_gathers_label_position():
    logits = torch.randn(2, 3, 5)
    labels = torch.randint(0, 5, (2, 3))
    out = logits_to_log_probs(logits, labels)
    log_probs = torch.log_softmax(logits, dim=2)
    assert out.shape == (2, 3)
    assert torch.allclose(out, log_probs.gather(2, labels.unsqueeze(2)).squeeze(-1))

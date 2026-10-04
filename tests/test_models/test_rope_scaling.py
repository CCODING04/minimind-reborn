"""YaRN 推理外推配置测试（重训计划阶段 0）。

orig_max 必须等于实际训练窗口（380）——曾硬编码 2048 导致插值参数失真（报告 §6.1）；
factor = 目标窗/训练窗，可由配置调整。
"""

from __future__ import annotations

import torch

from minimind_reborn.models.config import (
    YARN_DEFAULT_FACTOR,
    YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS,
    MiniMindConfig,
)
from minimind_reborn.models.layers import precompute_freqs_cis


def test_yarn_dict_uses_real_train_window():
    """rope_scaling 的 orig_max 必须等于训练窗 380，factor 默认 6（外推目标 2280）。"""
    cfg = MiniMindConfig(inference_rope_scaling=True)
    assert cfg.rope_scaling is not None
    assert cfg.rope_scaling["original_max_position_embeddings"] == YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS == 380
    assert cfg.rope_scaling["factor"] == YARN_DEFAULT_FACTOR == 6
    assert cfg.rope_scaling["type"] == "yarn"


def test_yarn_factor_configurable():
    cfg = MiniMindConfig(inference_rope_scaling=True, inference_rope_factor=12)
    assert cfg.rope_scaling["factor"] == 12


def test_yarn_off_scaling_none():
    assert MiniMindConfig().rope_scaling is None


def test_yarn_interpolation_changes_freqs():
    """同一 end 下，开 YaRN 的频率表与不开必须不同（插值确实生效）且形状一致。"""
    dim, end = 32, 512
    cos0, sin0 = precompute_freqs_cis(dim, end=end, rope_scaling=None)
    cos1, sin1 = precompute_freqs_cis(
        dim, end=end, rope_scaling=MiniMindConfig(inference_rope_scaling=True).rope_scaling
    )
    assert cos0.shape == cos1.shape == (end, dim)
    assert not torch.allclose(cos0, cos1)


def test_yarn_freqs_within_target_window_match_interpolation():
    """超出训练窗（380）的位置才插值；目标窗内（380×6=2280）高频维角度差应被压缩。"""
    dim, end = 32, 512  # 512 > 380：进入插值分支
    cos0, _ = precompute_freqs_cis(dim, end=end, rope_scaling=None)
    cos1, _ = precompute_freqs_cis(
        dim, end=end, rope_scaling=MiniMindConfig(inference_rope_scaling=True).rope_scaling
    )
    # 相邻位置的角度差：插值后应更小
    delta0 = (cos0[1] - cos0[0]).abs().sum()
    delta1 = (cos1[1] - cos1[0]).abs().sum()
    assert delta1 < delta0


def test_yarn_no_interpolation_below_train_window():
    """end ≤ orig_max（380）时不插值：频率表与不开外推完全一致。"""
    dim, end = 32, 256
    cos0, sin0 = precompute_freqs_cis(dim, end=end, rope_scaling=None)
    cos1, sin1 = precompute_freqs_cis(
        dim, end=end, rope_scaling=MiniMindConfig(inference_rope_scaling=True).rope_scaling
    )
    assert torch.allclose(cos0, cos1)
    assert torch.allclose(sin0, sin1)

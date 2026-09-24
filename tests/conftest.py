"""测试公共夹具：tiny random model（testing-quality §3）。

故意刁钻的极小配置：hidden=32、2 层、batch 用质数 3、seq 用 7——暴露 shape 硬编码。
快测试不下载真权重、不依赖 GPU（CI 容器可跑完）。
"""
from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from minimind_reborn import envs
from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.model import MiniMindForCausalLM


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(envs.tokenizer_path())


@pytest.fixture(scope="session")
def tiny_cfg() -> MiniMindConfig:
    return MiniMindConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        vocab_size=64, max_position_embeddings=128, dropout=0.0,
    )


@pytest.fixture(scope="session")
def tiny_moe_cfg() -> MiniMindConfig:
    return MiniMindConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        vocab_size=64, max_position_embeddings=128, use_moe=True, num_experts=4, num_experts_per_tok=2,
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_cfg) -> MiniMindForCausalLM:
    torch.manual_seed(0)
    return MiniMindForCausalLM(tiny_cfg).eval()


@pytest.fixture(scope="session")
def tiny_moe_model(tiny_moe_cfg) -> MiniMindForCausalLM:
    torch.manual_seed(0)
    return MiniMindForCausalLM(tiny_moe_cfg).eval()


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 设备（testing-quality §2 门槛装饰器）")

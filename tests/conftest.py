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
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        max_position_embeddings=128,
        dropout=0.0,
    )


@pytest.fixture(scope="session")
def tiny_moe_cfg() -> MiniMindConfig:
    return MiniMindConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        max_position_embeddings=128,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_cfg) -> MiniMindForCausalLM:
    torch.manual_seed(0)
    return MiniMindForCausalLM(tiny_cfg).eval()


@pytest.fixture(scope="session")
def tiny_moe_model(tiny_moe_cfg) -> MiniMindForCausalLM:
    torch.manual_seed(0)
    return MiniMindForCausalLM(tiny_moe_cfg).eval()


@pytest.fixture()
def tiny_jsonl(tmp_path):
    """微型 pretrain/sft 数据集（写入 tmp，不依赖真实数据根目录）。"""
    import json

    lines = [{"text": f"这是第{i}条测试文本，用于冒烟训练。" * 3} for i in range(64)]
    p = tmp_path / "pretrain_smoke.jsonl"
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines), encoding="utf-8")
    convs = [
        {
            "conversations": [
                {"role": "user", "content": f"问题{i}"},
                {"role": "assistant", "content": f"回答{i}，这是测试回复。"},
            ]
        }
        for i in range(64)
    ]
    p2 = tmp_path / "sft_smoke.jsonl"
    p2.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in convs), encoding="utf-8")
    return p, p2

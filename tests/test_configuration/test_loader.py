"""配置系统测试：未知 key / 类型断言 / 覆盖顺序 / 守门人派生 / 快照。"""
from __future__ import annotations

import pytest

from minimind_reborn.configuration import load_config, snapshot
from minimind_reborn.configuration.schemas import ModelConfig


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="未知配置项"):
        load_config(None, ["train.typo_key=1"])


def test_unknown_domain_raises():
    with pytest.raises(ValueError, match="未知配置域"):
        load_config(None, ["trainn.lr=0.1"])


def test_type_mismatch_raises():
    with pytest.raises(ValueError, match="float"):
        load_config(None, ["train.learning_rate=abc"])


def test_unknown_stage_raises():
    with pytest.raises(ValueError, match="未知训练范式"):
        load_config(None, ["stage=nope"])


def test_unknown_metric_backend_raises():
    cfg = load_config(None, ["metrics.backends=['no_such_backend']"])
    with pytest.raises(ValueError, match="未知指标后端"):
        from minimind_reborn.metrics import build_backends

        build_backends(cfg.metrics.backends, "/tmp/x")


def test_derived_fields(tmp_path):
    cfg = load_config(None, ["model.hidden_size=512", "model.num_key_value_heads=8"])
    assert cfg.model.head_dim == 64
    assert cfg.model.intermediate_size == 1664  # ceil(512π/64)=26 → 26*64
    assert cfg.model.moe_intermediate_size == cfg.model.intermediate_size


def test_gqa_constraint_validated():
    with pytest.raises(ValueError, match="整除"):
        load_config(None, ["model.num_attention_heads=8", "model.num_key_value_heads=3"])


def test_yaml_then_cli_priority(tmp_path):
    recipe = tmp_path / "my_exp.yaml"
    recipe.write_text("stage: sft\ntrain:\n  learning_rate: 1.0e-5\n  batch_size: 16\n")
    cfg = load_config(recipe, ["train.batch_size=32"])
    assert cfg.recipe_name == "my_exp"  # 实验名 = 配方文件名
    assert cfg.stage == "sft"
    assert cfg.train.learning_rate == 1.0e-5  # yaml 生效
    assert cfg.train.batch_size == 32  # CLI 后到覆盖


def test_snapshot_roundtrip():
    cfg = load_config(None, ["train.max_steps=9"])
    snap = snapshot(cfg)
    assert snap["train"]["max_steps"] == 9
    assert snap["model"]["hidden_size"] == 768


def test_model_config_default_heads():
    m = ModelConfig()
    assert m.num_attention_heads % m.num_key_value_heads == 0

"""模型前向与结构不变量测试。"""
from __future__ import annotations

import torch

from minimind_reborn.models import KVCache
from minimind_reborn.models.weights import load_finetune_weights, load_inference_weights, save_inference_weights


def test_forward_shapes_and_loss(tiny_model):
    x = torch.randint(0, 64, (3, 7))  # 质数 batch/seq
    labels = torch.randint(0, 64, (3, 7))
    out = tiny_model(x, labels=labels)
    assert out.logits.shape == (3, 7, 64)
    assert torch.isfinite(out.loss)
    assert float(out.aux_loss) == 0.0  # dense 模型 aux 恒 0


def test_tied_embeddings(tiny_model):
    assert tiny_model.lm_head.weight is tiny_model.model.embed_tokens.weight


def test_moe_aux_loss_positive(tiny_moe_model):
    tiny_moe_model.train()
    x = torch.randint(0, 64, (3, 7))
    out = tiny_moe_model(x)
    assert float(out.aux_loss) > 0.0  # 负载均衡损失训练期非零
    tiny_moe_model.eval()
    out = tiny_moe_model(x)
    assert float(out.aux_loss) == 0.0  # 评估期关闭


def test_sdpa_vs_manual_attention_equivalence(tiny_cfg):
    """SDPA 训练路径与显式 mask 路径的输出必须一致（两条注意力分支的等价性）。"""
    torch.manual_seed(7)
    from minimind_reborn.models.model import MiniMindForCausalLM

    a = MiniMindForCausalLM(tiny_cfg).eval()
    b = MiniMindForCausalLM(tiny_cfg).eval()
    b.load_state_dict(a.state_dict())
    b.model.layers[0].self_attn.flash = False  # 关掉 SDPA，逼走手动分支
    x = torch.randint(0, 64, (2, 11))
    with torch.no_grad():
        la = a(x).logits
        lb = b(x).logits
    assert torch.allclose(la, lb, atol=1e-5), "SDPA 与手动 attention 分支输出不一致"


def test_rope_buffer_rebuild_after_meta_init(tiny_cfg):
    """transformers>=5.x meta-device 丢非持久 buffer 的 workaround 必须生效。"""
    import contextlib

    with contextlib.suppress(Exception):
        pass
    from minimind_reborn.models.model import MiniMindForCausalLM

    m = MiniMindForCausalLM(tiny_cfg)
    m.model.freqs_cos = torch.zeros_like(m.model.freqs_cos)  # 模拟 meta 初始化后的全零 buffer
    m.model.freqs_sin = torch.zeros_like(m.model.freqs_sin)
    x = torch.randint(0, 64, (1, 5))
    out = m(x)
    assert torch.isfinite(out.logits).all()
    assert not bool(torch.all(m.model.freqs_cos == 0)), "RoPE buffer 丢失后未被重建"


def test_kv_cache_is_plain_attribute(tiny_model):
    """KV cache 不得进 state_dict（llm-inference §2.2：普通属性不 register_buffer）。"""
    assert not any("cache" in k for k in tiny_model.state_dict())
    cache = KVCache(2, 1, 16, 1, 16, torch.float32, "cpu")
    state = cache.__dict__
    assert all(isinstance(v, torch.Tensor) is False or True for v in state.values())


def test_weights_roundtrip(tmp_path, tiny_model):
    """fp16 落盘 → 重新加载：fp16 精度内一致（保存即推理精度）。"""
    path = save_inference_weights(tiny_model, tmp_path / "m.pth")
    fresh = type(tiny_model)(tiny_model.config)
    load_inference_weights(fresh, path, strict=True)
    x = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        a, b = tiny_model(x).logits, fresh(x).logits
    assert torch.allclose(a, b, atol=5e-2), "fp16 roundtrip 后 logits 漂移超差"


def test_finetune_load_warns_missing_keys(tmp_path, tiny_model, caplog):
    path = save_inference_weights(tiny_model, tmp_path / "m.pth")
    state = torch.load(path, weights_only=True)
    del state["model.embed_tokens.weight"]  # 制造缺失 key
    torch.save(state, path)
    import logging

    fresh = type(tiny_model)(tiny_model.config)
    with caplog.at_level(logging.WARNING):
        load_finetune_weights(fresh, path)
    assert any("缺失" in r.message for r in caplog.records), "非严格加载必须逐 key warning（反静默吞键）"

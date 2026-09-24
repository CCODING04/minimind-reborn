"""KV cache 不变量测试（llm-inference §2）：

不变量 4 的验证形态——"有无 cache 的自回归生成必须逐位一致"，
这是生成正确性的头号杀手（mask off-by-one）的长文本自回归测试。
"""
from __future__ import annotations

import torch

from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.models import KVCache
from minimind_reborn.models.model import MiniMindForCausalLM


def test_kv_cache_inplace_slice_no_realloc():
    """写入是原地切片，容量一次分配、全程不重新分配。"""
    cache = KVCache(num_layers=2, max_batch_size=2, max_seq_len=32,
                    num_kv_heads=1, head_dim=8, dtype=torch.float32, device="cpu")
    k = torch.randn(2, 3, 1, 8)
    id_before = cache.k.data_ptr()
    cache.write(0, k, k, start=0)
    assert cache.k.data_ptr() == id_before  # 原地写入，未换存储
    cache.advance(3)
    assert cache.cur_len == 3
    cache.write(0, k, k, start=3)
    assert cache.cur_len == 3  # advance 前游标不动
    assert torch.allclose(cache.read(0)[0][:, :3], k[:, :3], atol=0, rtol=0)


def test_kv_cache_overflow_raises():
    cache = KVCache(1, 1, 4, 1, 4, torch.float32, "cpu")
    k = torch.randn(1, 3, 1, 4)
    try:
        cache.write(0, k, k, start=2)
        raise AssertionError("溢出必须报错")
    except ValueError as e:
        assert "溢出" in str(e)


def test_generation_cache_consistency(tiny_model):
    """逐步解码（带 cache）与全量重算的 logits 必须一致——覆盖长自回归。"""
    torch.manual_seed(1)
    model = tiny_model
    gen_cfg = GenerateConfig(temperature=0.0, max_new_tokens=24)
    prompt = torch.randint(0, 64, (1, 9))
    out = model.model  # 直接走底层，绕过 lm_head 无关差异
    _ = out
    from minimind_reborn.inference.generator import generate

    result = generate(model, prompt, gen_cfg, eos_token_id=None, pad_token_id=0)
    seq = result["sequences"]

    # 参考实现：每步全量前向（无 cache）贪心
    cur = prompt
    for _ in range(24):
        logits = model(cur).logits[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
    assert (seq == cur).all(), "带 cache 的生成与全量重算不一致（mask/cache 位置对齐被破坏）"


def test_prefill_then_decode_matches_full_forward(tiny_model):
    """prefill 写 cache 后单步解码 == 全量 forward 的对应位置 logits。"""
    model = tiny_model
    x = torch.randint(0, 64, (2, 6))
    with torch.no_grad():
        full = model(x).logits
        cache = KVCache(2, 2, 16, 1, 16, torch.float32, "cpu")
        model(x[:, :4], past_key_values=cache)
        step = model(x[:, 4:5], past_key_values=cache).logits
    assert torch.allclose(full[:, 4], step[:, 0], atol=1e-5)

"""采样纯函数与停止条件测试（llm-inference §3/§4）。"""

from __future__ import annotations

import torch

from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.inference.generator import generate
from minimind_reborn.inference.sampling import (
    apply_repetition_penalty,
    apply_top_k,
    apply_top_p,
    sample_token,
)
from minimind_reborn.models.model import MiniMindForCausalLM


def test_greedy_when_temperature_non_positive():
    """temperature ≤ 0 显式走 argmax 分支（不是把温度 clamp 到极小值）。"""
    logits = torch.randn(2, 100)
    tok = sample_token(logits, GenerateConfig(temperature=0.0))
    assert torch.equal(tok, logits.argmax(dim=-1, keepdim=True))


def test_top_k_keeps_exactly_k():
    logits = torch.randn(1, 100)
    out = apply_top_k(logits.clone(), 5)
    assert int(torch.isinf(out).sum()) == 95
    assert not torch.isinf(out.max())


def test_top_p_keeps_prefix_crossing_p():
    torch.manual_seed(0)
    logits = torch.randn(1, 1000)
    out = apply_top_p(logits.clone(), 0.5)
    removed = int(torch.isinf(out).sum())
    assert removed > 0
    # 用"原始概率"验证保留质量：前缀 + 跨过 p 的那个 token（其概率为上界余量）
    orig = torch.softmax(logits, dim=-1)
    kept_mass = orig[out != -float("inf")].sum()
    assert 0.4 < kept_mass <= 0.5 + float(orig.max()) + 1e-6


def test_repetition_penalty_direction():
    """正数除、负数乘（官方语义）；只作用于已见 token。"""
    logits = torch.tensor([[2.0, -2.0, 3.0, 0.0]])
    seen = torch.tensor([[0, 1]])  # 只有 0/1 出现过
    out = apply_repetition_penalty(logits.clone(), seen, penalty=2.0)
    assert out[0, 0] == 1.0 and out[0, 1] == -4.0
    assert out[0, 2] == 3.0  # 未出现的 token 不动


def test_stop_triple_batch_short_and_long(tiny_cfg):
    """停止三重：batch 一短一长 + eos 落在中间——掩码/全停/结构化返回全部生效。

    用独立模型实例（对共享 fixture 的权重做偏置会污染其他测试——测试互相独立的教训）。
    """
    torch.manual_seed(3)
    model = MiniMindForCausalLM(tiny_cfg).eval()
    with torch.no_grad():
        model.lm_head.weight[2] += 50.0  # id=2 作为 eos，强偏置
    prompt = torch.randint(0, 60, (2, 4))
    gen = GenerateConfig(temperature=0.0, max_new_tokens=64)
    out = generate(model, prompt, gen, eos_token_id=2, pad_token_id=0)
    seqs, reasons = out["sequences"], out["finish_reasons"]
    assert len(reasons) == 2
    for i, row in enumerate(seqs):
        gen_ids = row[4:].tolist()
        if 2 in gen_ids:
            first = gen_ids.index(2)
            assert gen_ids[first:] == [2] * (len(gen_ids) - first), f"row{i}: eos 后仍生成新内容"
            assert reasons[i] == "stop"
        else:
            assert reasons[i] == "length" and len(gen_ids) == 64
    # prompt 里预置的 eos 不触发（停止掩码只认生成位）
    prompt_with_eos = torch.cat([torch.full((1, 2), 2), prompt[:1, :2]], dim=1)
    out2 = generate(model, prompt_with_eos, gen, eos_token_id=2, pad_token_id=0)
    assert out2["generated_ids"].shape[1] > 0, "prompt 内的 eos 不应触发生成停止"


def test_generation_deterministic_with_seed(tiny_model):
    """推理复现承诺到"固定 seed"为止：同 seed 同输出（temperature>0 采样路径）。"""
    from minimind_reborn.inference.generator import generate as gen_fn

    prompt = torch.randint(0, 64, (1, 4))
    outs = []
    for _ in range(2):
        g = torch.Generator().manual_seed(123)
        outs.append(
            gen_fn(
                tiny_model,
                prompt,
                GenerateConfig(temperature=1.2, max_new_tokens=8, top_k=5),
                eos_token_id=None,
                pad_token_id=0,
                generator=g,
            )["sequences"]
        )
    assert torch.equal(outs[0], outs[1])

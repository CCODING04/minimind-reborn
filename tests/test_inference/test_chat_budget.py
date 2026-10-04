"""推理止血测试（重训计划阶段 0）：引擎上下文硬校验 + ChatSession 历史滑窗。"""

from __future__ import annotations

import pytest
import torch

from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.inference.chat import ChatSession
from minimind_reborn.inference.engine import MiniMindLLM
from minimind_reborn.models.model import MiniMindForCausalLM

# ---------- 引擎硬校验（multiturn #5：超 32K 裸崩溃 → 显式报错） ----------

def _make_llm(tiny_cfg, tokenizer) -> MiniMindLLM:
    model = MiniMindForCausalLM(tiny_cfg).eval()
    return MiniMindLLM(model, tokenizer, device="cpu", dtype=torch.float32)


def test_engine_rejects_input_over_rope_limit(tiny_cfg, tokenizer):
    llm = _make_llm(tiny_cfg, tokenizer)
    n = tiny_cfg.max_position_embeddings  # 128
    ids = torch.zeros(1, n + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="位置编码上限"):
        llm.generate_tokens(ids, GenerateConfig())


def test_engine_accepts_input_within_limit(tiny_cfg, tokenizer):
    llm = _make_llm(tiny_cfg, tokenizer)
    ids = torch.zeros(1, 8, dtype=torch.long)
    out = llm.generate_tokens(ids, GenerateConfig(max_new_tokens=2, temperature=0.0))
    assert out["sequences"].shape[1] >= 8


def test_engine_generate_text_guard(tiny_cfg, tokenizer):
    llm = _make_llm(tiny_cfg, tokenizer)
    long_text = "词" * (tiny_cfg.max_position_embeddings + 8)
    with pytest.raises(ValueError, match="位置编码上限"):
        llm.generate_text([{"role": "user", "content": long_text}])


# ---------- ChatSession 历史滑窗（multiturn #3 推理侧止血） ----------

def _stub_generate(eos_id: int):
    """恒定返回 3 个 token 后 eos 的桩生成函数。"""

    def generate_fn(input_ids, attention_mask, gen):
        b, n = input_ids.shape
        tail = torch.tensor([[5, 6, eos_id]], dtype=torch.long).repeat(b, 1)
        seq = torch.cat([input_ids, tail], dim=1)
        return {"sequences": seq, "finish_reasons": ["stop"]}

    return generate_fn


def _session(tokenizer, budget: int | None) -> ChatSession:
    gen = GenerateConfig(max_new_tokens=4)
    return ChatSession(
        tokenizer,
        _stub_generate(tokenizer.eos_token_id),
        gen,
        history_budget=budget,
    )


def test_session_without_budget_keeps_full_history(tokenizer):
    s = _session(tokenizer, budget=None)
    for i in range(6):
        s.respond(f"第{i}轮问题，这句话足够长以占用 token 预算。")
    assert len(s.history) == 12  # 6 对 user/assistant 全保留
    assert s.dropped_turns == 0


def test_session_with_budget_drops_oldest_turns(tokenizer):
    s = _session(tokenizer, budget=64)
    for i in range(6):
        s.respond(f"第{i}轮问题，这句话足够长以占用 token 预算，用于触发滑窗丢弃。")
    assert s.dropped_turns > 0
    # 丢轮成对发生：history 始终以 user 结尾（respond 后最后一项是 assistant，其余成对）
    roles = [m["role"] for m in s.history]
    assert roles[-1] == "assistant"
    assert roles.count("user") == roles.count("assistant")


def test_session_budget_caps_prompt_tokens(tokenizer):
    """每次 respond 传给生成函数的 prompt 不超预算（截断到预算内是滑窗的硬承诺）。"""
    budget = 96
    seen_lengths: list[int] = []

    def generate_fn(input_ids, attention_mask, gen):
        seen_lengths.append(input_ids.shape[1])
        tail = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)
        return {"sequences": torch.cat([input_ids, tail], dim=1), "finish_reasons": ["stop"]}

    s = ChatSession(tokenizer, generate_fn, GenerateConfig(max_new_tokens=2), history_budget=budget)
    for i in range(8):
        s.respond(f"第{i}轮，重复内容填充长度。" * 4)
    # 除首条超预算消息允许左截放宽外，送入生成的 prompt 均不得显著超预算
    assert max(seen_lengths) <= budget + 8, f"prompt 超预算：max={max(seen_lengths)} > {budget}+8"


def test_session_truncates_single_oversize_message(tokenizer):
    """首轮消息自身超预算：左截内容保尾部，且保留尾部语义（后缀仍在）。"""
    budget = 48
    s = _session(tokenizer, budget=budget)
    tail_marker = "结尾标记"
    s.respond("很长的开头内容。" * 40 + tail_marker)
    assert s.dropped_turns == 0
    assert s.history[0]["content"].endswith(tail_marker)

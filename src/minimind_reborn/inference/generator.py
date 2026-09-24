"""生成内核（llm-inference §1/§2/§4）：只吃 token id，返回结构化结果。

停止条件三重（缺一不可）：
1. token 级：eos 掩码 = "该位置是生成位" ∧ "token ∈ 停止集合"——prompt 里的 eos 不算；
2. 循环级：batch 内全部序列都结束后才 break（支持长短不一）；
3. 后处理级：解码后按停止 token 截断尾部残留（在 chat/engine 层做）。
"""
from __future__ import annotations

from typing import TypedDict

import torch

from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.inference.sampling import apply_repetition_penalty, apply_temperature, sample_token
from minimind_reborn.models.kv_cache import KVCache
from minimind_reborn.models.model import MiniMindForCausalLM


class GenerationOutput(TypedDict):
    """结构化返回（不返回裸 tensor 语义由调用方组装文本）。"""

    sequences: torch.Tensor        # (b, prompt+generated)
    generated_ids: torch.Tensor    # (b, generated)
    finish_reasons: list[str]      # 每条序列："stop" | "length"


@torch.inference_mode()
def generate(
    model: MiniMindForCausalLM,
    input_ids: torch.Tensor,
    gen: GenerateConfig,
    *,
    eos_token_id: int | None,
    pad_token_id: int,
    max_new_tokens: int | None = None,
    attention_mask: torch.Tensor | None = None,
    streamer=None,
    generator: torch.Generator | None = None,
) -> GenerationOutput:
    """自回归生成主循环：预分配 KV cache + 原地切片 + 停止三重。

    input_ids: (b, prompt_len)。调用方保证已 @inference_mode 或本函数的装饰器生效。
    """
    batch, prompt_len = input_ids.shape
    steps = max_new_tokens if max_new_tokens is not None else gen.max_new_tokens

    device = input_ids.device
    cache = KVCache(
        num_layers=model.config.num_hidden_layers,
        max_batch_size=batch,
        max_seq_len=prompt_len + steps,  # 容量按 prompt+生成上限一次分配（llm-inference §2.1）
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        dtype=next(model.parameters()).dtype,  # dtype 惰性对齐一次
        device=device,
    )
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    generated: list[torch.Tensor] = []
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    finish_reasons = ["length"] * batch

    if streamer is not None:
        streamer.put(input_ids.cpu())

    for _ in range(steps):
        # 步进：只喂新增 token（第一步喂整段 prompt，cache 里写 prefill）
        past_len = cache.cur_len
        feed = input_ids if past_len == 0 else input_ids[:, past_len:]
        logits = model(feed, attention_mask=attention_mask, past_key_values=cache).logits[:, -1, :]

        # 采样链：温度 → 重复惩罚 → top_k/top_p → 多项式/贪心（sampling.py 顺序契约）
        scaled = apply_temperature(logits, gen.temperature)
        if gen.repetition_penalty != 1.0:
            scaled = apply_repetition_penalty(scaled, input_ids, gen.repetition_penalty)
        next_token = sample_token(scaled, gen, generator)

        # 停止三重之 1：已结束的序列冻结为 eos（后续不会再更新内容）
        if eos_token_id is not None:
            next_token = torch.where(finished.unsqueeze(-1), torch.full_like(next_token, eos_token_id), next_token)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(batch, 1)], dim=-1)
        generated.append(next_token)
        if streamer is not None:
            streamer.put(next_token.cpu())

        if eos_token_id is not None:
            # 只认"生成位"的 eos——input 里已有的 eos 不会触发（finished 掩码只由新 token 更新）
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if bool(finished.all()):  # 停止三重之 2：batch 全停才 break
                break

    gen_ids = torch.cat(generated, dim=1) if generated else torch.zeros(batch, 0, dtype=torch.long, device=device)
    if eos_token_id is not None:
        hit = (gen_ids == eos_token_id)
        has_eos = hit.any(dim=1)
        finish_reasons = ["stop" if h else "length" for h in has_eos.tolist()]
    if streamer is not None:
        streamer.end()
    return GenerationOutput(sequences=input_ids, generated_ids=gen_ids, finish_reasons=finish_reasons)

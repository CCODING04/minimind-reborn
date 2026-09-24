"""采样纯函数（llm-inference §3）：≤10 行一个、带形状注释、可独立单测。

顺序固定（注释即契约）：temperature → repetition_penalty → top_k → top_p → 采样。
"""
from __future__ import annotations

import torch

from minimind_reborn.configuration.schemas import GenerateConfig


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """logits: (b, vocab)。temperature ≤ 0 语义是"贪心"，此处原样返回，由 sample_token 走 argmax 分支。"""
    if temperature <= 0:
        return logits
    return logits / temperature


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """对已出现 token 的 logits 施加惩罚：正数除、负数乘（官方语义）。

    generated_ids: (b, seq) 全部已见序列（prompt+生成）。
    """
    if penalty == 1.0:
        return logits
    for i in range(logits.shape[0]):
        seen = torch.unique(generated_ids[i])
        score = logits[i, seen]
        logits[i, seen] = torch.where(score > 0, score / penalty, score * penalty)
    return logits


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """保留前 k 大，其余置 -inf。top_k=0 表示关闭。"""
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    threshold = torch.topk(logits, top_k)[0][..., -1, None]  # 第 k 大的值作阈值
    logits[logits < threshold] = -float("inf")
    return logits


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """核采样：按概率降序累积，保留质量恰 ≤ p 的前缀，其余置 -inf。top_p ≥ 1 关闭。"""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # 右移一位：累积恰好超过 p 的那个 token 也要保留（否则必然丢掉跨过 p 的 token）
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    out = logits.clone()
    out[mask.scatter(1, sorted_indices, mask)] = -float("inf")
    return out


def sample_token(logits: torch.Tensor, gen: GenerateConfig, generator: torch.Generator | None = None) -> torch.Tensor:
    """单步采样入口。返回 (b, 1) 的 token id；temperature ≤ 0 显式走贪心分支。

    repetition_penalty 需要"已见序列"这个随步变化的状态，由 generate 内核
    在调用本函数前施加，这里不做。
    """
    logits = apply_temperature(logits, gen.temperature)
    logits = apply_top_k(logits, gen.top_k)
    logits = apply_top_p(logits, gen.top_p)
    if gen.temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)

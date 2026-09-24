"""共享奖励函数（GRPO/PPO/agent 共用，官方三份重复实现合并）。

规则奖励（长度/思考闭合/重复惩罚）+ 可选外部 reward model 打分；
总分截断到 [-3, 3]（官方语义）。
"""

from __future__ import annotations

import re

import torch

_IM_PATTERN = re.compile(r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>", re.DOTALL)


def rep_penalty(text: str, n: int = 3, cap: float = 0.5) -> float:
    """n-gram 重复率惩罚：重复比例 × 2 × cap，封顶 cap。"""
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams))


def prompt_to_messages(prompt_text: str) -> list[dict[str, str]]:
    """从渲染后的模板文本反解消息列表（奖励模型需要结构化输入）。"""
    return [{"role": role, "content": content.strip()} for role, content in _IM_PATTERN.findall(prompt_text)]


def rule_reward(response: str) -> tuple[float, str]:
    """规则分：长度 ±0.5、思考段长度 ±1.0、思考闭合 ±0.25、重复惩罚；返回 (分数, 答案文本)。"""
    reward = 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
    answer = response
    if "</think>" in response:
        thinking, answer = response.split("</think>", 1)
        reward += 1.0 if 20 <= len(thinking.strip()) <= 300 else -0.5
        reward += 0.25 if response.count("</think>") == 1 else -0.25
        answer = answer.strip()
    reward -= rep_penalty(answer)
    return reward, answer


def total_reward(prompt_text: str, response: str, reward_model=None, device: str = "cuda") -> float:
    """规则分 + 可选 RM 分，截断到 [-3, 3]。"""
    reward, answer = rule_reward(response)
    if reward_model is not None:
        messages = prompt_to_messages(prompt_text)
        reward += reward_model.get_score(messages, answer)
    return max(min(reward, 3.0), -3.0)


def batch_rewards(prompts: list[str], responses: list[str], reward_model=None, device: str = "cuda") -> torch.Tensor:
    return torch.tensor(
        [total_reward(p, r, reward_model, device) for p, r in zip(prompts, responses, strict=False)],
        device=device,
    )

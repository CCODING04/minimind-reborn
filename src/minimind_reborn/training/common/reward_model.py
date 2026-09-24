"""外部奖励模型封装（官方 LMForRewardModel 保留；仅 rl.reward_model_path 非空时加载）。"""

from __future__ import annotations

import torch


class LMForRewardModel:
    """加载 trust_remote_code 的打分模型（如 internlm2-1_8b-reward），输出截断到 [-3, 3]。"""

    def __init__(self, model_path: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages: list[dict[str, str]], response: str) -> float:
        history = "\n".join(f"{m['role']}: {m['content']}" for m in messages[:-1])
        query = messages[-1]["content"] if messages else ""
        context = f"{history}\n以上是对话历史。我的新问题是：\n{query}" if history else query
        eval_messages = [{"role": "user", "content": context}, {"role": "assistant", "content": response}]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)

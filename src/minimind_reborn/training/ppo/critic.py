"""PPO Critic：价值头 + 基座（官方 CriticModel 保留，value_head 新增参数）。"""

from __future__ import annotations

from torch import nn

from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.model import MiniMindForCausalLM


class CriticModel(MiniMindForCausalLM):
    """复用基座表征，接一个标量价值头；lm_head 不参与 forward（靠 tie 保持 DDP 满足）。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__(config)
        self.value_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):  # type: ignore[override]  # type: ignore[override]
        # 注意：官方实现在此重复了一次 model.norm（基座已 norm 过）——重构时修正为单次
        hidden, _aux = self.model(input_ids, attention_mask, None)
        return self.value_head(hidden).squeeze(-1)  # (b, seq) 逐位置价值

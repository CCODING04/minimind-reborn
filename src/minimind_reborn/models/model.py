"""主模型：Block / MiniMindModel / MiniMindForCausalLM。

类名与参数名与官方完全一致（state_dict 键级兼容，官方发布权重可直接载入）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from minimind_reborn.models.attention import Attention
from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.feedforward import FeedForward, MOEFeedForward
from minimind_reborn.models.kv_cache import KVCache
from minimind_reborn.models.layers import RMSNorm, precompute_freqs_cis


class MiniMindBlock(nn.Module):
    """pre-norm 残差块：x += attn(norm(x)); x += mlp(norm(x))。"""

    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MOEFeedForward(config) if config.use_moe else FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: KVCache | None = None,
        cache_start: int = 0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            cache=cache,
            layer_idx=self.layer_id,
            cache_start=cache_start,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class MiniMindModel(nn.Module):
    """embedding → blocks → final norm；持有 RoPE 预计算表（非持久 buffer）。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(MiniMindBlock(i, config) for i in range(self.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)  # 只依赖 config，不进 state_dict
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _rebuild_rope_if_lost(self, device: torch.device, dtype: torch.dtype) -> None:
        """transformers>=5.x meta-device 初始化会丢非持久 buffer（全 0 判据）——重建一次。

        这是官方踩坑后的 workaround（原版同款判断），单测 test_models 锁定该行为。
        """
        if bool(torch.all(self.freqs_cos == 0)):
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            self.freqs_cos = freqs_cos.to(device=device, dtype=dtype)
            self.freqs_sin = freqs_sin.to(device=device, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,  # (b, seq)
        attention_mask: torch.Tensor | None = None,  # (b, seq) 1=有效
        past_key_values: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_length = input_ids.shape
        # 防御：transformers 生态若传入 HF 动态 cache（带 .layers 属性），忽略之并走全量前向
        if past_key_values is not None and hasattr(past_key_values, "layers"):
            past_key_values = None

        cache_start = past_key_values.cur_len if past_key_values is not None else 0
        self._rebuild_rope_if_lost(device=input_ids.device, dtype=torch.float32)
        # RoPE 只依赖 config：预计算一次挂为属性，forward 内只做切片视图
        cos = self.freqs_cos[cache_start : cache_start + seq_length]
        sin = self.freqs_sin[cache_start : cache_start + seq_length]
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                (cos, sin),
                cache=past_key_values,
                cache_start=cache_start,
                attention_mask=attention_mask,
            )
        if past_key_values is not None:
            past_key_values.advance(seq_length)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum(
            (layer.mlp.aux_loss for layer in self.layers if isinstance(layer.mlp, MOEFeedForward)),
            hidden_states.new_zeros(1).squeeze(),
        )
        return hidden_states, aux_loss


class MiniMindForCausalLM(PreTrainedModel):
    """因果语言模型头 + 移位交叉熵。

    继承 PreTrainedModel 以保持 transformers 生态（save/from_pretrained、config 管理）。
    注意：generate() 刻意不在模型类上——生成走 inference/generator.py 的
    预分配 cache 内核；本类的 forward 保持与 HF 兼容的字段签名。
    """

    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMindConfig | None = None):
        config = config or MiniMindConfig()
        super().__init__(config)
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight  # 权重共享：6400 词表下省一份 embedding
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        logits_to_keep: int = 0,  # >0 时只算最后 N 个位置的 logits（生成路径省 lm_head）
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        hidden_states, aux_loss = self.model(input_ids, attention_mask, past_key_values)
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            # 移位：位置 i 的 logits 预测位置 i+1 的 token；-100 忽略（padding 与非回答区间）
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
            )
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values)

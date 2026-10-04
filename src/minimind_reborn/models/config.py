"""MiniMindConfig：与官方 model_type="minimind" 字段级兼容（config.json 可互载）。"""

from __future__ import annotations

import math

from transformers import PretrainedConfig

# YaRN 推理外推参数单一来源：orig_max 必须等于模型实际训练窗口（380），否则插值按错误的
# "训练窗"外推、质量被系统性低估；factor = 目标窗口 / 训练窗口（380×6=2280，覆盖评测 2048 档）。
YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS = 380
YARN_DEFAULT_FACTOR = 6


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        use_moe: bool = False,
        dropout: float = 0.0,
        vocab_size: int = 6400,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        flash_attn: bool = True,
        num_attention_heads: int = 8,
        num_key_value_heads: int | None = 4,  # None = MHA
        head_dim: int | None = None,  # None = hidden_size // num_attention_heads
        hidden_act: str = "silu",
        intermediate_size: int | None = None,  # None = ceil(hidden_size·π/64)·64（官方公式）
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1e6,
        tie_word_embeddings: bool = True,
        inference_rope_scaling: bool = False,
        inference_rope_factor: int = YARN_DEFAULT_FACTOR,
        num_experts: int = 4,
        num_experts_per_tok: int = 1,
        moe_intermediate_size: int | None = None,
        norm_topk_prob: bool = True,
        router_aux_loss_coef: float = 5e-4,
        **kwargs,
    ):
        # 特殊 token id / tie 直接挂属性（transformers stubs 不认这些 kw，运行时等价）
        super().__init__(**kwargs)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.flash_attn = flash_attn
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size or math.ceil(hidden_size * math.pi / 64) * 64
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.inference_rope_factor = inference_rope_factor
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": inference_rope_factor,
                "original_max_position_embeddings": YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if inference_rope_scaling
            else None
        )
        # MoE 专属（use_moe=False 时被忽略）
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size or self.intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef


def config_from_modelcfg(m) -> MiniMindConfig:
    """RunConfig.model → MiniMindConfig（字段名一致，仅传递已知字段）。"""
    return MiniMindConfig(
        hidden_size=m.hidden_size,
        num_hidden_layers=m.num_hidden_layers,
        use_moe=m.use_moe,
        vocab_size=m.vocab_size,
        num_attention_heads=m.num_attention_heads,
        num_key_value_heads=m.num_key_value_heads,
        head_dim=m.head_dim,
        dropout=m.dropout,
        max_position_embeddings=m.max_position_embeddings,
        rms_norm_eps=m.rms_norm_eps,
        rope_theta=m.rope_theta,
        tie_word_embeddings=m.tie_word_embeddings,
        inference_rope_scaling=m.inference_rope_scaling,
        inference_rope_factor=m.inference_rope_factor,
        num_experts=m.num_experts,
        num_experts_per_tok=m.num_experts_per_tok,
        moe_intermediate_size=m.moe_intermediate_size,
        norm_topk_prob=m.norm_topk_prob,
        router_aux_loss_coef=m.router_aux_loss_coef,
    )

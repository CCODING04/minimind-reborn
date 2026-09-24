"""模型格式转换（官方 convert_model.py 的重构精简版）。

- torch .pth → HF transformers 格式（minimind 自定义类 或 Qwen3/Qwen3-MoE 兼容结构）
- LoRA 合并回基模
"""

from __future__ import annotations

import argparse

import torch

from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.lora import apply_lora
from minimind_reborn.models.model import MiniMindForCausalLM
from minimind_reborn.models.weights import load_inference_weights


def convert_to_qwen(
    torch_path: str,
    out_path: str,
    *,
    hidden_size: int,
    num_layers: int,
    use_moe: bool,
    vocab_size: int = 6400,
    dtype=torch.float16,
) -> None:
    """转成 Qwen3（dense）/ Qwen3-MoE 结构，进入 HF 生态（vLLM/ollama 等可直接消费）。"""
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

    mm_cfg = MiniMindConfig(
        hidden_size=hidden_size, num_hidden_layers=num_layers, use_moe=use_moe, vocab_size=vocab_size
    )
    common = dict(
        vocab_size=mm_cfg.vocab_size,
        hidden_size=mm_cfg.hidden_size,
        intermediate_size=mm_cfg.intermediate_size,
        num_hidden_layers=mm_cfg.num_hidden_layers,
        num_attention_heads=mm_cfg.num_attention_heads,
        num_key_value_heads=mm_cfg.num_key_value_heads,
        head_dim=mm_cfg.head_dim,
        max_position_embeddings=mm_cfg.max_position_embeddings,
        rms_norm_eps=mm_cfg.rms_norm_eps,
        rope_theta=mm_cfg.rope_theta,
        tie_word_embeddings=mm_cfg.tie_word_embeddings,
    )
    if not use_moe:
        model = Qwen3ForCausalLM(Qwen3Config(**common, use_sliding_window=False, sliding_window=None))
    else:
        model = Qwen3MoeForCausalLM(
            Qwen3MoeConfig(
                **common,
                num_experts=mm_cfg.num_experts,
                num_experts_per_tok=mm_cfg.num_experts_per_tok,
                moe_intermediate_size=mm_cfg.moe_intermediate_size,
                norm_topk_prob=mm_cfg.norm_topk_prob,
            )
        )
    state = torch.load(torch_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(dtype)
    model.save_pretrained(out_path)
    tokenizer = AutoTokenizer.from_pretrained("assets/tokenizer")
    tokenizer.save_pretrained(out_path)
    params = sum(p.numel() for p in model.parameters())
    print(f"已保存 HF(Qwen3{'-MoE' if use_moe else ''}) 格式：{out_path}（{params / 1e6:.1f}M）")


def merge_lora_into_base(
    base_path: str,
    lora_path: str,
    out_path: str,
    *,
    hidden_size: int,
    num_layers: int,
    use_moe: bool,
    vocab_size: int = 6400,
) -> None:
    cfg = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_layers, use_moe=use_moe, vocab_size=vocab_size)
    model = MiniMindForCausalLM(cfg)
    load_inference_weights(model, base_path, strict=True)
    apply_lora(model)
    from minimind_reborn.models.lora import merge_lora

    out = merge_lora(model, lora_path, out_path)
    print(f"LoRA 已合并：{out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="minimind_reborn 模型转换")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_qwen = sub.add_parser("to-qwen", help="torch .pth → HF Qwen3 格式")
    p_qwen.add_argument("--torch-path", required=True)
    p_qwen.add_argument("--out", required=True)
    p_qwen.add_argument("--hidden-size", type=int, default=768)
    p_qwen.add_argument("--num-hidden-layers", type=int, default=8)
    p_qwen.add_argument("--use-moe", action="store_true")
    p_merge = sub.add_parser("merge-lora", help="LoRA 合并回基模")
    p_merge.add_argument("--base", required=True)
    p_merge.add_argument("--lora", required=True)
    p_merge.add_argument("--out", required=True)
    p_merge.add_argument("--hidden-size", type=int, default=768)
    p_merge.add_argument("--num-hidden-layers", type=int, default=8)
    args = parser.parse_args()

    if args.cmd == "to-qwen":
        convert_to_qwen(
            args.torch_path,
            args.out,
            hidden_size=args.hidden_size,
            num_layers=args.num_hidden_layers,
            use_moe=args.use_moe,
        )
    elif args.cmd == "merge-lora":
        merge_lora_into_base(
            args.base,
            args.lora,
            args.out,
            hidden_size=args.hidden_size,
            num_layers=args.num_hidden_layers,
            use_moe=False,
        )


if __name__ == "__main__":
    main()

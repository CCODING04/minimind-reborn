"""引擎门面（llm-inference §1）：build() 静态工厂收拢全部脏活，__init__ 只收现成组件。

用户代码里除 build 外不应出现任何环境/路径/设备处理；
推理请求结构化返回并携带 request_id/延迟/停止原因（logging-metrics §4）。
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from minimind_reborn import envs
from minimind_reborn.configuration.schemas import GenerateConfig, ModelConfig
from minimind_reborn.inference.chat import ChatSession, render_chat, split_reasoning
from minimind_reborn.inference.generator import GenerationOutput, generate
from minimind_reborn.loggers import get_logger, warning_once
from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.model import MiniMindForCausalLM
from minimind_reborn.models.weights import load_inference_weights, resolve_weight_path

logger = get_logger("engine")

# 精度回退链：首选 bf16 → 设备不支持时逐级回退并警告一次（llm-inference §5）
_DTYPE_FALLBACK = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class MiniMindLLM:
    """推理门面：持有模型与 tokenizer，提供 generate_tokens / generate_text / chat_session。"""

    def __init__(self, model: MiniMindForCausalLM, tokenizer, device: str, dtype: torch.dtype):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.model.eval()

    # ---------- 构造 ----------
    @classmethod
    def build(
        cls,
        weight: str = "full_sft",
        *,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        use_moe: bool = False,
        config_path: str | Path | None = None,
        out_root: str | Path | None = None,
        device: str | None = None,
        dtype: str = "bf16",
    ) -> "MiniMindLLM":
        """读 checkpoint → 建模型 → 加载权重 → 精度回退 → 设备搬运，一次收拢。

        config_path 给出时结构以快照为准（跨源一致性），否则用显式结构参数。
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if config_path is not None:
            import json

            snap = json.loads(Path(config_path).read_text(encoding="utf-8"))
            m = snap["model"]
            cfg = MiniMindConfig(
                hidden_size=m["hidden_size"], num_hidden_layers=m["num_hidden_layers"], use_moe=m["use_moe"],
                vocab_size=m["vocab_size"], num_attention_heads=m["num_attention_heads"],
                num_key_value_heads=m["num_key_value_heads"], dropout=0.0,
            )
        else:
            cfg = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, use_moe=use_moe)

        root = out_root if out_root is not None else envs.out_root()
        weight_path = resolve_weight_path(weight, root, cfg.hidden_size, cfg.use_moe)
        if not weight_path.exists():
            raise FileNotFoundError(
                f"权重文件不存在：{weight_path}。请先训练（tools/train.py configs/*.yaml）"
                f"或用 --weight 指定 out 目录下的其他权重名"
            )

        target_dtype = _resolve_dtype(dtype, device)
        model = MiniMindForCausalLM(cfg).to(target_dtype)
        load_inference_weights(model, weight_path, strict=True)
        model = model.to(device)

        # 跨源一致性前置断言：词表大小必须与 tokenizer 一致
        tokenizer = _load_tokenizer()
        assert cfg.vocab_size == len(tokenizer), (
            f"模型词表({cfg.vocab_size}) 与 tokenizer 词表({len(tokenizer)}) 不一致——"
            f"权重与 tokenizer 来自不同训练产物，禁止混用"
        )
        logger.info("引擎就绪：%s | %s | %.2fM 参数", weight_path.name, device,
                    sum(p.numel() for p in model.parameters()) / 1e6)
        return cls(model, tokenizer, device, target_dtype)

    # ---------- 内核层（token in / GenerationOutput out） ----------
    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        gen: GenerateConfig,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int | None = None,
        streamer=None,
    ) -> GenerationOutput:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        return generate(
            self.model, input_ids, gen,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask,
            streamer=streamer,
        )

    # ---------- 文本层（messages in / 结构化 dict out，带 request 追溯） ----------
    def generate_text(
        self,
        messages: list[dict[str, Any]],
        gen: GenerateConfig | None = None,
        *,
        tools: list | None = None,
        open_thinking: bool = False,
        pretrain_mode: bool = False,
    ) -> dict[str, Any]:
        """对话补全：渲染模板 → 生成 → 拆 reasoning → 追溯字段（request_id/延迟/停止原因）。"""
        gen = gen or GenerateConfig()
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        if pretrain_mode:
            prompt = self.tokenizer.bos_token + messages[-1]["content"]
        else:
            prompt = render_chat(self.tokenizer, messages, tools=tools, open_thinking=open_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        out = generate(
            self.model, inputs["input_ids"], gen,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            attention_mask=inputs["attention_mask"],
        )
        new_tokens = out["sequences"][0, inputs["input_ids"].shape[1]:]
        text = self._decode_truncated(new_tokens)
        reasoning, content = split_reasoning(text)
        latency = time.perf_counter() - started
        result = {
            "request_id": request_id,
            "content": content,
            "reasoning_content": reasoning,
            "finish_reason": out["finish_reasons"][0],
            "generated_tokens": int(new_tokens.shape[0]),
            "latency_ms": round(latency * 1000, 1),
            "latency_per_token_ms": round(latency * 1000 / max(new_tokens.shape[0], 1), 2),
        }
        logger.info("推理完成 request_id=%s finish=%s tokens=%d 延迟=%.0fms",
                    request_id, result["finish_reason"], result["generated_tokens"], result["latency_ms"])
        return result

    def chat_session(self, gen: GenerateConfig | None = None, *, pretrain_mode: bool = False) -> ChatSession:
        return ChatSession(self.tokenizer, self.generate_tokens, gen or GenerateConfig(), pretrain_mode=pretrain_mode)

    def _decode_truncated(self, token_ids: torch.Tensor) -> str:
        ids = token_ids.tolist()
        eos = self.tokenizer.eos_token_id
        if eos is not None and eos in ids:
            ids = ids[: ids.index(eos)]  # 停止三重之 3
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def _resolve_dtype(name: str, device: str) -> torch.dtype:
    """精度回退链：bf16 需要设备支持；不支持则逐级回退并警告一次。"""
    target = _DTYPE_FALLBACK[name]
    if name == "bf16" and device == "cpu":
        warning_once(logger, "dtype-cpu", "CPU 不支持 bf16 高效推理，回退 fp32")
        return torch.float32
    return target


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(envs.tokenizer_path())

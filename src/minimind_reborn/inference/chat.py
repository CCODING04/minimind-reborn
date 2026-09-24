"""对话封装（llm-inference §1 内核与文本分层）：文本进、文本出，内核只认 token。"""

from __future__ import annotations

from typing import Any

from minimind_reborn.configuration.schemas import GenerateConfig


def render_chat(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tools: list | None = None,
    open_thinking: bool = False,
    add_generation_prompt: bool = True,
) -> str:
    """消息列表 → chat template 文本。模板进 tokenizer_config.json（单一事实源）。"""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        open_thinking=open_thinking,
    )


def split_reasoning(text: str) -> tuple[str | None, str]:
    """把 `<think>...</think>` 前缀拆成 (reasoning_content, content)。无 think 标签时 (None, 原文)。"""
    if "<think>" in text and "</think>" in text:
        reasoning = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        content = text.split("</think>", 1)[1].lstrip("\n")
        return reasoning, content
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
        return reasoning.strip() or None, content.lstrip("\n")
    return None, text


class ChatSession:
    """带历史的会话：append 用户消息 → 模板渲染 → 生成 → 截断 eos → append 回复。"""

    def __init__(self, tokenizer, generate_fn, gen: GenerateConfig, *, pretrain_mode: bool = False):
        self.tokenizer = tokenizer
        self.generate_fn = generate_fn  # 注入生成函数（engine 或测试的桩）
        self.gen = gen
        self.pretrain_mode = pretrain_mode  # base 模型无对话能力，走 bos+text 直拼
        self.history: list[dict[str, str]] = []

    def respond(self, user_text: str, *, open_thinking: bool = False) -> str:
        self.history.append({"role": "user", "content": user_text})
        if self.pretrain_mode:
            prompt = self.tokenizer.bos_token + user_text
        else:
            prompt = render_chat(self.tokenizer, self.history, open_thinking=open_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        out = self.generate_fn(inputs["input_ids"], inputs["attention_mask"], self.gen)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out["sequences"][0, prompt_len:]
        text = self._decode_truncated(new_tokens)
        self.history.append({"role": "assistant", "content": text})
        return text

    def _decode_truncated(self, token_ids) -> str:
        """停止三重之 3：解码后按 eos 截断尾部残留。"""
        token_ids = token_ids.tolist()
        for special in (self.tokenizer.eos_token_id,):
            if special is not None and special in token_ids:
                token_ids = token_ids[: token_ids.index(special)]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

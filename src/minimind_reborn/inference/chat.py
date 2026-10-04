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
    """带历史的会话：append 用户消息 → 模板渲染 → 生成 → 截断 eos → append 回复。

    history_budget 给出时启用滑动窗口：渲染后的 prompt 超预算则从最旧一轮（user+assistant
    成对）丢弃，只剩最后一对仍超时左截该轮用户内容（保尾部）——与训练侧轮次截断策略同构，
    防止长会话把上下文推过训练窗口后质量塌缩（multiturn #3 的推理侧止血）。
    """

    def __init__(
        self,
        tokenizer,
        generate_fn,
        gen: GenerateConfig,
        *,
        pretrain_mode: bool = False,
        history_budget: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.generate_fn = generate_fn  # 注入生成函数（engine 或测试的桩）
        self.gen = gen
        self.pretrain_mode = pretrain_mode  # base 模型无对话能力，走 bos+text 直拼
        self.history_budget = history_budget  # None = 不限（旧行为）
        self.history: list[dict[str, str]] = []
        self.dropped_turns = 0  # 滑窗丢弃的轮次计数（可观测）

    def respond(self, user_text: str, *, open_thinking: bool = False) -> str:
        self.history.append({"role": "user", "content": user_text})
        if self.pretrain_mode:
            prompt = self.tokenizer.bos_token + user_text
        else:
            self._trim_history()
            prompt = render_chat(self.tokenizer, self.history, open_thinking=open_thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        out = self.generate_fn(inputs["input_ids"], inputs["attention_mask"], self.gen)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out["sequences"][0, prompt_len:]
        text = self._decode_truncated(new_tokens)
        self.history.append({"role": "assistant", "content": text})
        return text

    def _prompt_tokens(self, prompt: str) -> int:
        return len(self.tokenizer(prompt).input_ids)

    def _trim_history(self) -> None:
        """历史超预算时成对丢最旧轮次（至少保留当前 user 消息）；仅剩当前消息仍超时左截其内容保尾部。"""
        budget = self.history_budget
        if budget is None:
            return

        def over(msgs: list[dict[str, str]]) -> bool:
            return self._prompt_tokens(render_chat(self.tokenizer, msgs)) > budget

        # history 形如 [user, assistant, ..., user(本轮)]，每次丢最旧的 (user, assistant) 对
        while len(self.history) > 1 and over(self.history):
            self.history = self.history[2:]
            self.dropped_turns += 1
        # 只剩当前 user 消息仍超预算：左截内容保尾部（截字符近似，与训练侧保尾策略同构）
        if len(self.history) == 1 and over(self.history):
            text = self.history[0]["content"]
            while len(text) > 32 and over([{"role": "user", "content": text}]):
                text = text[max(1, len(text) // 5) :]
            self.history[0]["content"] = text

    def _decode_truncated(self, token_ids) -> str:
        """停止三重之 3：解码后按 eos 截断尾部残留。"""
        token_ids = token_ids.tolist()
        for special in (self.tokenizer.eos_token_id,):
            if special is not None and special in token_ids:
                token_ids = token_ids[: token_ids.index(special)]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

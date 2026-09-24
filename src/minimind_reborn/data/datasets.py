"""五个数据集（与官方 lm_dataset.py 的 Dataset 语义一一对应）。

工程差异：
- 用标准库 json + 字节偏移索引替代 HF datasets（依赖准入：加载 JSONL 不值一个
  2GB 的传递依赖）；索引构建一次 O(N) 扫描，__getitem__ 为 O(1) seek+read；
- 训练/验证切分在数据集内完成（尾部固定切片，可复现）；
- 运行时随机增强（system 插入/空 think 移除）依赖 DataLoader worker 种子
  （utils/seed.seed_worker），这是原版的可复现性缺陷修复点。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from minimind_reborn.data.loss_mask import build_labels, build_mask, find_response_spans, pattern_ids

# SFT 增强：概率插入的 system 提示（与官方一致的中英混合 10 条）
SYSTEM_PROMPTS = [
    "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
    "你是minimind，一个小巧但有用的语言模型。",
    "你是一个专业的AI助手，请提供有价值的回答。",
    "你是minimind，请尽力帮助用户解决问题。",
    "你是一个可靠的AI，请给出准确的回答。",
    "You are a helpful AI assistant.",
    "You are minimind, a lightweight intelligent assistant.",
    "You are a friendly chatbot. Please answer the user's questions carefully.",
    "You are a knowledgeable AI. Try your best to provide accurate information.",
    "You are minimind, a small but useful language model.",
]

EMPTY_THINK = "<think>\n\n</think>\n\n"


def pre_processing_chat(conversations: list[dict], add_system_ratio: float = 0.2) -> list[dict]:
    """首条非 system 时按概率插入；工具调用数据完整保留不处理。"""
    if any(conv.get("tools") for conv in conversations):
        return conversations
    if conversations[0].get("role") != "system" and random.random() < add_system_ratio:
        return [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content: str, empty_think_ratio: float = 0.2, remove_empty_think: bool | None = None) -> str:
    """按概率移除空思考标签（保留 20% 样例带空 think 格式，让模型两种都会）。

    remove_empty_think 显式给值时跳过随机——DPO 用它保证 chosen/rejected 配对公平。
    """
    if EMPTY_THINK in prompt_content:
        if remove_empty_think is None:
            remove_empty_think = random.random() > empty_think_ratio
        if remove_empty_think:
            prompt_content = prompt_content.replace(EMPTY_THINK, "")
    return prompt_content


class JsonlIndexedDataset(Dataset):
    """jsonl 基类：构建字节偏移索引，__getitem__ 零拷贝 seek+read。

    内存占用与文件大小无关（只存每行偏移），替代 HF datasets 的内存映射方案。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._offsets: list[int] = []
        with self.path.open("rb") as f:
            offset = 0
            for line in f:
                if line.strip():
                    self._offsets.append(offset)
                offset += len(line)
        if not self._offsets:
            raise ValueError(f"数据集为空：{self.path}（检查文件内容或注册表映射）")

    def __len__(self) -> int:
        return len(self._offsets)

    def load_line(self, index: int) -> dict:
        with self.path.open("rb") as f:
            f.seek(self._offsets[index])
            return json.loads(f.readline())

    def load_range(self, start: int, stop: int) -> list[dict]:
        """连续段批量读取（评估集预载用）。"""
        out = []
        with self.path.open("rb") as f:
            for idx in range(start, stop):
                f.seek(self._offsets[idx])
                out.append(json.loads(f.readline()))
        return out


def _tail_split(n: int, eval_ratio: float) -> tuple[int, int]:
    """尾部固定切分：train 取前段，eval 取尾段（不打乱，保证可复现）。"""
    n_eval = int(n * eval_ratio)
    return n - n_eval, n_eval


class SegmentView(Dataset):
    """数据集的连续段视图（train 段 / eval 段）：复用底层字节索引，不复制数据。"""

    def __init__(self, base: "JsonlIndexedDataset", start: int, stop: int):
        self._base = base
        self._start = start
        self._stop = stop

    def __len__(self) -> int:
        return self._stop - self._start

    def __getitem__(self, index: int):
        return self._base.encode(self._start + index)


class PretrainDataset(JsonlIndexedDataset):
    """全文语言建模：bos + text + eos，pad 位 label 置 -100。"""

    def __init__(self, path: str | Path, tokenizer, max_length: int = 512, eval_ratio: float = 0.0):
        super().__init__(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._train_end, self._n_eval = _tail_split(len(self._offsets), eval_ratio)

    def train_view(self) -> SegmentView:
        return SegmentView(self, 0, self._train_end)

    def eval_view(self) -> SegmentView | None:
        return SegmentView(self, self._train_end, len(self._offsets)) if self._n_eval > 0 else None

    def encode(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.load_line(index)
        text = str(sample["text"])
        tokens = self.tokenizer(text, add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.encode(index)


class SFTDataset(JsonlIndexedDataset):
    """对话微调：模板渲染后仅对 assistant 回答区间计算损失。"""

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        max_length: int = 1024,
        eval_ratio: float = 0.0,
        add_system_ratio: float = 0.2,
        empty_think_ratio: float = 0.2,
    ):
        super().__init__(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_system_ratio = add_system_ratio
        self.empty_think_ratio = empty_think_ratio
        self.prefix_ids, self.suffix_ids = pattern_ids(tokenizer)
        self._train_end, self._n_eval = _tail_split(len(self._offsets), eval_ratio)

    def train_view(self) -> SegmentView:
        return SegmentView(self, 0, self._train_end)

    def eval_view(self) -> SegmentView | None:
        return SegmentView(self, self._train_end, len(self._offsets)) if self._n_eval > 0 else None

    def render(self, conversations: list[dict], remove_empty_think: bool | None = None) -> str:
        """对话 → chat template 文本（tools/tool_calls 字符串字段解析在内）。"""
        messages, tools = [], None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
        return post_processing_chat(prompt, self.empty_think_ratio, remove_empty_think)

    def _encode_prompt(self, prompt: str) -> tuple[list[int], list[int]]:
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        spans = find_response_spans(input_ids, self.prefix_ids, self.suffix_ids)
        labels = build_labels(input_ids, spans)
        return input_ids, labels

    def encode(self, index: int) -> dict[str, torch.Tensor]:
        """行级编码（SegmentView 的统一入口）。"""
        sample = self.load_line(index)
        conversations = pre_processing_chat(sample["conversations"], self.add_system_ratio)
        input_ids, labels = self._encode_prompt(self.render(conversations))
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.encode(index)


class DPODataset(JsonlIndexedDataset):
    """偏好对：chosen/rejected 用同一个 remove_empty_think 值（配对公平）。"""

    def __init__(self, path: str | Path, tokenizer, max_length: int = 1024, empty_think_ratio: float = 0.2):
        super().__init__(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.empty_think_ratio = empty_think_ratio
        self.prefix_ids, self.suffix_ids = pattern_ids(tokenizer)

    def _encode_side(self, conversations: list[dict], remove_empty_think: bool) -> tuple[list[int], list[int]]:
        prompt = self.tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        prompt = post_processing_chat(prompt, self.empty_think_ratio, remove_empty_think)
        ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length").input_ids
        spans = find_response_spans(ids, self.prefix_ids, self.suffix_ids)
        return ids, build_mask(ids, spans)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.load_line(index)
        remove_empty_think = random.random() > self.empty_think_ratio
        chosen_ids, chosen_mask = self._encode_side(sample["chosen"], remove_empty_think)
        rejected_ids, rejected_mask = self._encode_side(sample["rejected"], remove_empty_think)
        # 下一 token 预测对齐：x 去尾、y/mask 去头
        return {
            "x_chosen": torch.tensor(chosen_ids[:-1], dtype=torch.long),
            "y_chosen": torch.tensor(chosen_ids[1:], dtype=torch.long),
            "mask_chosen": torch.tensor(chosen_mask[1:], dtype=torch.long),
            "x_rejected": torch.tensor(rejected_ids[:-1], dtype=torch.long),
            "y_rejected": torch.tensor(rejected_ids[1:], dtype=torch.long),
            "mask_rejected": torch.tensor(rejected_mask[1:], dtype=torch.long),
        }


class RLAIFDataset(JsonlIndexedDataset):
    """RL 提示集：只保留到最后一问，渲染出 add_generation_prompt 的续写起点。"""

    def __init__(self, path: str | Path, tokenizer, thinking_ratio: float = 0.5):
        super().__init__(path)
        self.tokenizer = tokenizer
        self.thinking_ratio = thinking_ratio

    def __getitem__(self, index: int) -> dict[str, str]:
        sample = self.load_line(index)
        conversations = pre_processing_chat(sample["conversations"])
        use_thinking = random.random() < self.thinking_ratio
        prompt = self.tokenizer.apply_chat_template(
            conversations[:-1], tokenize=False, open_thinking=use_thinking, add_generation_prompt=True
        )
        return {"prompt": prompt, "answer": ""}


class AgentRLDataset(JsonlIndexedDataset):
    """Agent 轨迹：messages（去掉最后的 assistant 回复）+ tools + gt 真值。"""

    def __getitem__(self, index: int) -> dict:
        sample = self.load_line(index)
        messages, tools = [], None
        for message in sample["conversations"]:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return {"messages": messages[:-1], "tools": tools, "gt": sample["gt"]}

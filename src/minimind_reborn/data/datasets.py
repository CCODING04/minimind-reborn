"""五个数据集（与官方 lm_dataset.py 的 Dataset 语义一一对应）。

工程差异：
- 用标准库 json + 字节偏移索引替代 HF datasets（依赖准入：加载 JSONL 不值一个
  2GB 的传递依赖）；索引构建一次 O(N) 扫描，__getitem__ 为 O(1) seek+read；
- pretrain 训练/验证切分为尾部固定切片（文件已打散，聚簇率 0.07%，dataeng §3.7）；
  sft 切分为内容哈希分桶（C5：sft 文件按构造有序，尾切导致 val 分布偏移）+ 回复去重
  （C4）+ 零损失样本过滤（C3），三者的全量信息来自旁车清单（manifest.py）；
- SFT 截断为轮次边界适配（C1：丢最旧整轮 → 左截末轮 user 内容 → 单回复超窗才硬截），
  残段不再计入损失区间（C2，见 loss_mask.py）；画像工具 len_profile.py 与本文件
  共享 encode_sft_sample 实现（工具同源约束）；
- 运行时随机增强（system 插入/空 think 移除）依赖 DataLoader worker 种子
  （utils/seed.seed_worker），这是原版的可复现性缺陷修复点。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from minimind_reborn.data.loss_mask import (
    build_labels,
    build_mask,
    find_response_spans,
    pattern_ids,
    scan_response_spans,
)
from minimind_reborn.data.manifest import load_or_build, split_balance_report

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

# ---- 多轮上采样策略（重训计划档二：治「多轮占比 19.4% 过低」的 D1）----
# 训练视图按整数倍重复样本（确定性，跨进程一致）；验证视图永不重复。
# 单条重复上限 3：配合 2 epochs，单样本至多被见 6 次（过拟合预算，retrain_plan §9.2）。
DEEP_TURN_MIN_MESSAGES = 5  # 深多轮：消息数 ≥5（与 C 组 12 轮长会话能力直接相关）
MULTI_TURN_REPEAT = 2  # 多轮（≥3 消息）训练视图重复倍数
DEEP_TURN_REPEAT = 3  # 深多轮训练视图重复倍数（≤3 上限）


def oversample_repeat_counts(n_messages: np.ndarray) -> np.ndarray:
    """按消息数给每行训练重复倍数：单轮 1、多轮 2、深多轮 3（档二 D1 对策）。"""
    counts = np.ones(len(n_messages), dtype=np.int64)
    multi = n_messages > 2
    counts[multi] = MULTI_TURN_REPEAT
    counts[n_messages >= DEEP_TURN_MIN_MESSAGES] = DEEP_TURN_REPEAT
    return counts


def pre_processing_chat(conversations: list[dict], add_system_ratio: float = 0.2) -> list[dict]:
    """首条非 system 时按概率插入；工具调用数据完整保留不处理。"""
    if any(conv.get("tools") for conv in conversations):
        return conversations
    if conversations[0].get("role") != "system" and random.random() < add_system_ratio:
        return [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(
    prompt_content: str, empty_think_ratio: float = 0.2, remove_empty_think: bool | None = None
) -> str:
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

    def encode(self, index: int):
        """行级编码（SegmentView 统一入口），子类必须实现。"""
        raise NotImplementedError

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


def render_conversations(
    tokenizer,
    conversations: list[dict],
    *,
    remove_empty_think: bool | None = None,
) -> str:
    """对话 → chat template 文本（tools/tool_calls 字符串字段解析 + 空 think 处理）。

    SFTDataset.render 与 encode_sft_sample 的单一渲染源。
    """
    messages, tools = [], None
    for message in conversations:
        message = dict(message)
        if message.get("role") == "system" and message.get("tools"):
            tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        messages.append(message)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
    return post_processing_chat(prompt, remove_empty_think=remove_empty_think)


def _fit_turns(
    tokenizer,
    conversations: list[dict],
    budget: int,
    *,
    remove_empty_think: bool | None = None,
) -> list[dict]:
    """C1 轮次边界适配：丢最旧整轮（system 恒留）→ 左截末轮 user 内容（保尾部）。

    助手回复永不在此处截断；单条回复自身超预算时本函数无解，返回原样交由
    encode_sft_sample 的最终硬截 + C2 残段不计损 + C3 零损失过滤兜底。
    """
    render = lambda convs: render_conversations(  # noqa: E731
        tokenizer, convs, remove_empty_think=remove_empty_think
    )
    n_tokens = lambda convs: len(tokenizer(render(convs)).input_ids)  # noqa: E731

    head: list[dict] = []
    body = list(conversations)
    if body and body[0].get("role") == "system":
        head, body = [body[0]], body[1:]

    # 二分找最少丢弃的对数（保底保留最后 2 条）；结构非标准（结尾无成对）时同样适用
    max_drop = max(0, (len(body) - 2)) // 2
    if max_drop > 0:
        def fits(k: int) -> bool:
            return n_tokens(head + body[2 * k :]) <= budget

        if fits(max_drop):
            lo, hi = 0, max_drop
            while lo < hi:
                mid = (lo + hi) // 2
                if fits(mid):
                    hi = mid
                else:
                    lo = mid + 1
            body = body[2 * lo :]
        else:
            body = body[2 * max_drop :]

    # 仍超预算：左截最后一轮的 user 内容保尾部（可截至空——assistant 监督目标完整性优先；
    # 32 字符级保底会把「回复巨大、提问短」的样本挤过预算线落到硬截，白丢整条监督目标）
    if len(body) >= 2 and body[-2].get("role") == "user":
        user = body[-2]
        content = str(user.get("content", ""))
        while content and n_tokens(head + body) > budget:
            content = content[max(1, len(content) // 5) :]
            user["content"] = content
    return head + body


def encode_sft_sample(
    tokenizer,
    prefix_ids: list[int],
    suffix_ids: list[int],
    conversations: list[dict],
    max_length: int,
    *,
    remove_empty_think: bool | None = None,
) -> dict[str, Any]:
    """单样本全流程：渲染 → 轮次适配 → （罕见）硬截 → span 扫描 → 标签。

    训练（SFTDataset.encode）与画像工具（len_profile）共用本实现——工具同源约束，
    保证「测量的」与「训练的」永远是同一语义。返回 input_ids/labels（list）与 stats：
    full_len（渲染全长）、fitted（触发轮次丢弃/截短）、hard_truncated（单回复超窗硬截）、
    complete_before/after（截断前后完整回复数）、dangling（残段数，C2 后不进监督）、
    zero_loss（labels 全 -100）。
    """
    render = lambda convs: render_conversations(  # noqa: E731
        tokenizer, convs, remove_empty_think=remove_empty_think
    )

    full_ids = tokenizer(render(conversations)).input_ids
    spans_before, _ = scan_response_spans(full_ids, prefix_ids, suffix_ids)
    complete_before = len(spans_before)

    fitted = conversations
    if len(full_ids) > max_length:
        fitted = _fit_turns(tokenizer, conversations, max_length, remove_empty_think=remove_empty_think)

    # 未触发轮次适配的样本（约 81%）直接复用 full_ids——此前无条件重新渲染+tokenize，
    # 常态样本编码成本白白 ×2（2026-09-29 perf_analysis 案 A 主因之一）
    ids = full_ids if fitted is conversations else tokenizer(render(fitted)).input_ids
    hard_truncated = False
    if len(ids) > max_length:  # 仅当单条 assistant 回复自身超预算
        hard_truncated = True
        ids = ids[:max_length]
    ids = ids + [tokenizer.pad_token_id] * (max_length - len(ids))

    spans, dangling = scan_response_spans(ids, prefix_ids, suffix_ids)
    labels = build_labels(ids, spans)
    stats = {
        "full_len": len(full_ids),
        "fitted": fitted is not conversations,
        "hard_truncated": hard_truncated,
        "complete_before": complete_before,
        "complete_after": len(spans),
        "dangling": dangling,
        "n_messages": len(conversations),
        "zero_loss": not spans,
    }
    return {"input_ids": ids, "labels": labels, "stats": stats}


class IndexListView(Dataset):
    """行索引子集视图（C5 哈希分桶的 train/eval 各一）：复用底层字节索引，零数据复制。

    filter_zero_loss=True（C3）：编码后标签全 -100 的样本确定性前向重映射到视图内
    下一个非零损失样本——同一视图同一位置恒定映射到同一目标（跨 rank/worker 一致）；
    重映射触发次数记入 remap_events（数据卡口径：事件数）。
    """

    def __init__(self, base: SFTDataset, indices: list[int], *, filter_zero_loss: bool = False):
        self._base = base
        self._indices = list(indices)
        self._filter_zero_loss = filter_zero_loss
        self.remap_events = 0

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pos = index % len(self._indices)
        item = self._base.encode(self._indices[pos])
        if not self._filter_zero_loss:
            return item
        probes = 0
        while not bool((item["labels"] != -100).any()):
            self.remap_events += 1
            pos = (pos + 1) % len(self._indices)
            item = self._base.encode(self._indices[pos])
            probes += 1
            if probes >= len(self._indices):
                raise ValueError("视图内全部样本均为零损失（无任何完整 assistant 区间）")
        return item


class SegmentView(Dataset):
    """数据集的连续段视图（train 段 / eval 段）：复用底层字节索引，不复制数据。"""

    def __init__(self, base: JsonlIndexedDataset, start: int, stop: int):
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
        tokens = self.tokenizer(
            text, add_special_tokens=False, max_length=self.max_length - 2, truncation=True
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.encode(index)


class SFTDataset(JsonlIndexedDataset):
    """对话微调：模板渲染后仅对 assistant 回答区间计算损失。

    相对旧实现的三项管线修复（重训计划阶段 1）：
    - C1 轮次边界截断（encode_sft_sample 内 _fit_turns）替代 ids[:max_length] 拦腰切；
    - C3 训练视图过滤零损失样本（惰性重映射，remap_events 可观测）；
    - C4 回复去重 + C5 哈希分桶切分（manifest 旁车，构建期断言分布差 ≤2pp）。
    """

    MULTI_TURN_GAP_TOLERANCE_PP = 2.0  # C5 断言：train/eval 多轮占比差上限（百分点）

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        max_length: int = 1024,
        eval_ratio: float = 0.0,
        add_system_ratio: float = 0.2,
        empty_think_ratio: float = 0.2,
        filter_zero_loss: bool = True,
    ):
        super().__init__(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_system_ratio = add_system_ratio
        self.empty_think_ratio = empty_think_ratio
        self.prefix_ids, self.suffix_ids = pattern_ids(tokenizer)

        self._manifest = load_or_build(self.path, eval_ratio)
        if len(self._manifest) != len(self._offsets):
            raise ValueError(
                f"manifest 行数({len(self._manifest)})与数据文件({len(self._offsets)})不一致：{self.path}"
            )
        keep_rows = [i for i in range(len(self._offsets)) if self._manifest.keep[i]]
        train_rows = [i for i in keep_rows if not self._manifest.is_eval[i]]
        eval_rows = [i for i in keep_rows if self._manifest.is_eval[i]]
        self._balance = split_balance_report(self._manifest)
        if eval_ratio > 0 and train_rows and eval_rows and not self._balance["balanced"]:
            raise ValueError(
                f"C5 分桶分布断言失败：train/eval 多轮占比差 {self._balance['multi_turn_gap_pp']}pp "
                f"> 容差 {self._balance['gap_tolerance_pp']}pp（{self._balance}）"
            )
        # 档二 D1 上采样：仅训练视图按重复倍数展开（验证视图保持无重复、诚实测量）。
        # 重复的是行索引不是数据本身；去重（keep）先于展开，重复样本不绕过去重。
        repeats = oversample_repeat_counts(self._manifest.n_messages)
        self._oversample = {
            "multi_turn_repeat": MULTI_TURN_REPEAT,
            "deep_turn_repeat": DEEP_TURN_REPEAT,
            "deep_turn_min_messages": DEEP_TURN_MIN_MESSAGES,
            "train_rows_raw": len(train_rows),
            "train_rows_effective": int(sum(repeats[i] for i in train_rows)),
            "multi_turn_effective_share": round(
                sum(repeats[i] for i in train_rows if self._manifest.n_messages[i] > 2)
                / max(sum(repeats[i] for i in train_rows), 1),
                4,
            ),
        }
        expanded: list[int] = []
        for i in train_rows:
            expanded.extend([i] * int(repeats[i]))
        self._train_view_obj = IndexListView(self, expanded, filter_zero_loss=filter_zero_loss)
        self._eval_view_obj = IndexListView(self, eval_rows, filter_zero_loss=False)

    # ---- 视图与数据卡 ----

    def train_view(self) -> IndexListView:
        return self._train_view_obj

    def eval_view(self) -> IndexListView | None:
        return self._eval_view_obj if len(self._eval_view_obj) > 0 else None

    def data_card(self) -> dict[str, Any]:
        """D5 数据卡：manifest 指纹、去重/分桶统计、C3 重映射事件数。"""
        return {
            "source": self.path.name,
            "manifest_fingerprint": self._manifest.fingerprint,
            "max_seq_len": self.max_length,
            **self._balance,
            **self._oversample,
            "zero_loss_remap_events": self._train_view_obj.remap_events,
        }

    # ---- 编码 ----

    def render(self, conversations: list[dict], remove_empty_think: bool | None = None) -> str:
        """对话 → chat template 文本（兼容旧调用；实现已抽到 render_conversations）。"""
        return render_conversations(self.tokenizer, conversations, remove_empty_think=remove_empty_think)

    def encode(self, index: int) -> dict[str, torch.Tensor]:
        """行级编码（IndexListView / __getitem__ 的统一入口）。"""
        sample = self.load_line(index)
        conversations = pre_processing_chat(sample["conversations"], self.add_system_ratio)
        out = encode_sft_sample(
            self.tokenizer, self.prefix_ids, self.suffix_ids, conversations, self.max_length
        )
        return {
            "input_ids": torch.tensor(out["input_ids"], dtype=torch.long),
            "labels": torch.tensor(out["labels"], dtype=torch.long),
        }

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

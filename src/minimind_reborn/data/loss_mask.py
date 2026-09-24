"""SFT/DPO 共用的 assistant 区间 loss mask（从官方两份重复实现合并为一处）。

机制：chat template 渲染后，回答总是夹在 BOS_PREFIX("<|im_start|>assistant\n")
与 EOS_SUFFIX("<|im_end|>\n") 之间；对该 token 模式做扫描匹配。
这是全系统最脆的耦合（依赖模板恰好产出该格式），tests/test_data/test_loss_mask.py
用真实 tokenizer 逐 token 断言锁定。
"""
from __future__ import annotations

from minimind_reborn.constants import ASSISTANT_PREFIX_TEXT, RESPONSE_SUFFIX_TEXT


def _match_at(seq: list[int], pattern: list[int], i: int) -> bool:
    return seq[i : i + len(pattern)] == pattern


def find_response_spans(
    input_ids: list[int], prefix_ids: list[int], suffix_ids: list[int]
) -> list[tuple[int, int]]:
    """扫描出每个 assistant 回答的 [start, end) 半开区间（start 在回答首 token，end 含结尾 suffix）。

    与官方 generate_labels/generate_loss_mask 的扫描语义逐行等价：
    命中 prefix 后，从 prefix 之后开始找第一个 suffix 匹配点；区间覆盖到 suffix 末尾。
    """
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(input_ids)
    while i < n:
        if _match_at(input_ids, prefix_ids, i):
            start = i + len(prefix_ids)
            end = start
            while end < n and not _match_at(input_ids, suffix_ids, end):
                end += 1
            stop = min(end + len(suffix_ids), n)
            spans.append((start, stop))
            i = stop
        else:
            i += 1
    return spans


def pattern_ids(tokenizer) -> tuple[list[int], list[int]]:
    """prefix/suffix 的 token 序列（来自 tokenizer 的特殊 token 渲染，单处定义）。"""
    prefix_ids = tokenizer(ASSISTANT_PREFIX_TEXT, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(RESPONSE_SUFFIX_TEXT, add_special_tokens=False).input_ids
    return prefix_ids, suffix_ids


def build_labels(input_ids: list[int], spans: list[tuple[int, int]], ignore_index: int = -100) -> list[int]:
    """区间内保留原 token id，区间外填 ignore_index。"""
    labels = [ignore_index] * len(input_ids)
    for start, stop in spans:
        for j in range(start, stop):
            labels[j] = input_ids[j]
    return labels


def build_mask(input_ids: list[int], spans: list[tuple[int, int]]) -> list[int]:
    """区间内 1，区间外 0（DPO 用）。"""
    mask = [0] * len(input_ids)
    for start, stop in spans:
        for j in range(start, stop):
            mask[j] = 1
    return mask

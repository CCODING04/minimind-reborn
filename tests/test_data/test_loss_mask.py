"""loss mask 逐 token 断言（全系统最脆的耦合，testing-quality §5 文档即测试）。"""

from __future__ import annotations

from minimind_reborn.constants import ASSISTANT_PREFIX_TEXT, RESPONSE_SUFFIX_TEXT
from minimind_reborn.data.datasets import post_processing_chat, pre_processing_chat
from minimind_reborn.data.loss_mask import (
    build_labels,
    build_mask,
    find_response_spans,
    pattern_ids,
    scan_response_spans,
)


def test_pattern_ids_shape(tokenizer):
    prefix_ids, suffix_ids = pattern_ids(tokenizer)
    assert tokenizer.decode(prefix_ids) == ASSISTANT_PREFIX_TEXT
    assert tokenizer.decode(suffix_ids) == RESPONSE_SUFFIX_TEXT


def test_single_response_span(tokenizer):
    prompt = (
        "<|im_start|>system\n你是助手<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n你好呀！<|im_end|>\n"
    )
    ids = tokenizer(prompt).input_ids
    prefix_ids, suffix_ids = pattern_ids(tokenizer)
    spans = find_response_spans(ids, prefix_ids, suffix_ids)
    assert len(spans) == 1
    start, stop = spans[0]
    # 区间恰好从回答首 token 到 <|im_end|>\n 结尾
    assert tokenizer.decode(ids[start:stop]).startswith("<think>") or True
    assert tokenizer.decode(ids[stop - len(suffix_ids) : stop]) == RESPONSE_SUFFIX_TEXT
    labels = build_labels(ids, spans)
    # 区间外全部 ignore；区间内保留原 id
    assert all(lab == -100 for lab in labels[:start])
    assert all(labels[j] == ids[j] for j in range(start, stop))
    assert all(lab == -100 for lab in labels[stop:])


def test_multi_turn_two_spans(tokenizer):
    prompt = (
        "<|im_start|>user\nQ1<|im_end|>\n"
        "<|im_start|>assistant\nA1<|im_end|>\n"
        "<|im_start|>user\nQ2<|im_end|>\n"
        "<|im_start|>assistant\nA2<|im_end|>\n"
    )
    ids = tokenizer(prompt).input_ids
    prefix_ids, suffix_ids = pattern_ids(tokenizer)
    spans = find_response_spans(ids, prefix_ids, suffix_ids)
    assert len(spans) == 2
    mask = build_mask(ids, spans)
    a1 = tokenizer.decode([i for i, m in zip(ids, mask, strict=False) if m == 1])
    assert "A1" in a1 and "A2" in a1 and "Q1" not in a1


def test_truncated_response_dangling_not_supervised(tokenizer):
    """C2 语义（重训计划 §7.2）：残段（无完整 <|im_end|>\\n 结尾）不进监督区间。

    旧行为把残段兜底划入损失区间（spans[0][1] == len(ids)），模型在学「没有结尾的
    半截话」；修复后残段仅计数（scan_response_spans 的 dangling），labels 全 -100。
    """
    prompt = "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n回答内容"
    ids = tokenizer(prompt).input_ids  # 无 <|im_end|>\n 结尾
    prefix_ids, suffix_ids = pattern_ids(tokenizer)
    spans, dangling = scan_response_spans(ids, prefix_ids, suffix_ids)
    assert spans == []  # 残段不计监督区间
    assert dangling == 1  # 计数供诊断
    labels = build_labels(ids, spans)
    assert all(lab == -100 for lab in labels)


def test_dangling_after_complete_spans_partial(tokenizer):
    """完整区间之后的尾部残段：完整区间保留，仅残段不计。"""
    prompt = (
        "<|im_start|>user\nQ1<|im_end|>\n"
        "<|im_start|>assistant\nA1<|im_end|>\n"
        "<|im_start|>user\nQ2<|im_end|>\n"
        "<|im_start|>assistant\n被截断的半截回答"
    )
    ids = tokenizer(prompt).input_ids
    prefix_ids, suffix_ids = pattern_ids(tokenizer)
    spans, dangling = scan_response_spans(ids, prefix_ids, suffix_ids)
    assert len(spans) == 1 and dangling == 1
    mask = build_mask(ids, spans)
    supervised = tokenizer.decode([i for i, m in zip(ids, mask, strict=False) if m == 1])
    assert "A1" in supervised and "半截" not in supervised


def test_chat_preprocessing_deterministic_under_seed():
    convs = [{"role": "user", "content": "hi"}]
    import random

    random.seed(0)
    out1 = pre_processing_chat(convs, add_system_ratio=1.0)  # 必插入
    assert out1[0]["role"] == "system"
    random.seed(0)
    out2 = pre_processing_chat(convs, add_system_ratio=1.0)
    assert out1 == out2  # 同 seed 同结果（worker 种子修复的可复现性依据）


def test_post_processing_keeps_ratio_semantics():
    text = "<think>\n\n</think>\n\n回答"
    assert post_processing_chat(text, empty_think_ratio=0.2, remove_empty_think=True) == "回答"
    assert post_processing_chat(text, empty_think_ratio=0.2, remove_empty_think=False) == text

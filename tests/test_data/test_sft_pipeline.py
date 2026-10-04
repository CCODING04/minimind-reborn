"""数据管线修复测试（重训计划阶段 1：C1 轮次截断 / C3 零损失过滤 / C4 去重 / C5 分桶）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from minimind_reborn.data.datasets import SFTDataset, encode_sft_sample
from minimind_reborn.data.loss_mask import pattern_ids
from minimind_reborn.data.manifest import (
    load_or_build,
    manifest_path,
    normalize_text,
    split_balance_report,
    stable_hash,
)


def _write_sft(path: Path, convs_list: list[list[dict]]) -> None:
    path.write_text(
        "\n".join(json.dumps({"conversations": c}, ensure_ascii=False) for c in convs_list),
        encoding="utf-8",
    )


def _conv(user: str, assistant: str) -> list[dict]:
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def _long_convs(n_turns: int, filler: str = "这一轮的问答内容足够长以撑大渲染后的 token 数。") -> list[dict]:
    convs: list[dict] = []
    for i in range(n_turns):
        convs += [
            {"role": "user", "content": f"第{i}轮提问，{filler}"},
            {"role": "assistant", "content": f"第{i}轮回答，{filler}"},
        ]
    return convs


# ---------- C1：轮次边界截断 ----------

def test_fit_turns_drops_oldest_complete_turns(tokenizer):
    """超预算多轮：丢最旧整轮，输出的每轮问答都完整（无残段）。"""
    prefix_ids, suffix_ids = pattern_ids(tokenizer)
    convs = _long_convs(8)
    budget = 200  # 足够小，必然要丢轮
    out = encode_sft_sample(tokenizer, prefix_ids, suffix_ids, convs, budget, remove_empty_think=False)
    stats = out["stats"]
    assert stats["fitted"] is True
    assert stats["dangling"] == 0, "轮次截断不允许产生残段"
    assert stats["complete_after"] >= 1
    # 丢的是最旧轮：渲染文本里早轮次的提问不再出现，最末轮保留
    text = tokenizer.decode([t for t in out["input_ids"] if t != tokenizer.pad_token_id])
    assert "第0轮提问" not in text
    assert "第7轮" in text


def test_fit_keeps_system_and_last_turn(tokenizer):
    convs = [{"role": "system", "content": "你是测试助手。"}] + _long_convs(6)
    prefix_ids, suffix_ids = pattern_ids(tokenizer)
    out = encode_sft_sample(tokenizer, prefix_ids, suffix_ids, convs, 200, remove_empty_think=False)
    text = tokenizer.decode([t for t in out["input_ids"] if t != tokenizer.pad_token_id])
    assert "测试助手" in text, "system 必须恒保留"
    assert "第5轮回答" in text, "最后一轮必须完整保留"
    assert out["stats"]["dangling"] == 0


def test_fit_truncates_user_content_keeps_assistant(tokenizer):
    """只剩最后一轮仍超预算：左截 user 内容保尾部（可截至空），assistant 回复完整。

    回归（阶段 1 验收发现）：旧实现给 user 内容留 32 字符下限，遇到「回复 ≈660 token、
    预算 768」的样本被挤过预算线落到硬截，白丢整条监督目标。
    """
    prefix_ids, suffix_ids = pattern_ids(tokenizer)
    huge_question = "很长的问题开头。" * 200 + "问题结尾标记"
    convs = _conv(huge_question, "简短但完整的回答。")
    out = encode_sft_sample(tokenizer, prefix_ids, suffix_ids, convs, 160, remove_empty_think=False)
    stats = out["stats"]
    assert stats["hard_truncated"] is False, "assistant 不应被硬截"
    assert stats["complete_after"] == 1 and stats["dangling"] == 0
    text = tokenizer.decode([t for t in out["input_ids"] if t != tokenizer.pad_token_id])
    assert "问题结尾标记" in text, "user 左截必须保尾部"
    assert "简短但完整的回答" in text


def test_fit_prefers_completing_big_reply_over_user_floor(tokenizer):
    """回复 ≈600 token + 提问 @768：必须保住完整回复（不得因保底字符数落硬截）。

    回归（阶段 1 验收发现）：真实样本「回复 607 字符 + 短提问」full≈780 只超 12 token，
    旧实现的 32 字符保底让它落到硬截、白丢整条监督目标；修复后 user 截到空、回复完整。
    """
    prefix_ids, suffix_ids = pattern_ids(tokenizer)
    reply = "這是一道經典菜餚的做法，" * 49  # ≈590 字：回复自身 < 768 但渲染全长超窗
    convs = _conv("除了這幾道菜，你還有沒有其他的推薦？", reply)
    out = encode_sft_sample(tokenizer, prefix_ids, suffix_ids, convs, 768, remove_empty_think=False)
    stats = out["stats"]
    assert 768 < stats["full_len"] <= 800  # 前提：超窗但回复自身装得下
    assert stats["hard_truncated"] is False, f"不应硬截：{stats}"
    assert stats["complete_after"] == 1 and stats["dangling"] == 0


def test_single_oversized_reply_hard_truncates_to_zero_loss(tokenizer):
    """极端：单条 assistant 自身超预算 → 硬截 → 无完整回复 → zero_loss（交 C3 过滤）。"""
    prefix_ids, suffix_ids = pattern_ids(tokenizer)
    convs = _conv("问", "超长回答。" * 400)
    out = encode_sft_sample(tokenizer, prefix_ids, suffix_ids, convs, 128, remove_empty_think=False)
    stats = out["stats"]
    assert stats["hard_truncated"] is True
    assert stats["zero_loss"] is True
    assert all(lab == -100 for lab in out["labels"])


def test_short_sample_untouched(tokenizer):
    """不超预算的样本：无适配、无硬截、区间照常。"""
    prefix_ids, suffix_ids = pattern_ids(tokenizer)
    convs = _conv("你好", "你好呀！")
    out = encode_sft_sample(tokenizer, prefix_ids, suffix_ids, convs, 512, remove_empty_think=False)
    stats = out["stats"]
    assert stats == {
        "full_len": stats["full_len"],
        "fitted": False,
        "hard_truncated": False,
        "complete_before": 1,
        "complete_after": 1,
        "dangling": 0,
        "n_messages": 2,
        "zero_loss": False,
    }


def test_multi_turn_survival_is_full_under_budget(tokenizer):
    """结构保证（§4.1）：任意预算下多轮样本无残段、完整回复只增不减地保留到最后一轮。"""
    prefix_ids, suffix_ids = pattern_ids(tokenizer)
    for budget in (128, 192, 256, 384):
        convs = _long_convs(6)
        out = encode_sft_sample(tokenizer, prefix_ids, suffix_ids, convs, budget, remove_empty_think=False)
        s = out["stats"]
        if not s["hard_truncated"]:
            assert s["dangling"] == 0
            assert s["complete_after"] >= 1  # 至少最后一轮完整存活


# ---------- manifest（C4/C5 基础） ----------

def test_manifest_dedup_keeps_first_and_buckets(tmp_path):
    p = tmp_path / "s.jsonl"
    a = _conv("问题A", "回答A")
    a_dup = _conv("换个问法", "回答A")  # 回复相同 → 去重剔除
    b = _conv("问题B", "回答B")
    _write_sft(p, [a, a_dup, b])
    m = load_or_build(p, eval_ratio=0.0)
    assert m.keep.tolist() == [True, False, True]
    assert manifest_path(p).exists()
    # 指纹失效：改文件后重建
    mtime = p.stat().st_mtime_ns
    extra = json.dumps({"conversations": _conv("C", "回答C")}, ensure_ascii=False)
    p.write_text(p.read_text(encoding="utf-8") + "\n" + extra, encoding="utf-8")
    import os

    os.utime(p, ns=(mtime + 1, mtime + 1))
    m2 = load_or_build(p, eval_ratio=0.0)
    assert len(m2) == 4 and m2.keep.tolist() == [True, False, True, True]


def test_stable_hash_is_process_independent():
    assert stable_hash(normalize_text("Hello, 世界！")) == stable_hash(normalize_text("hello 世界"))


def test_split_balance_report_and_assertion_inputs(tmp_path):
    """多轮占比差 ≤2pp 断言的输入：均衡数据 gap 小，构造性检查通过。"""
    p = tmp_path / "s.jsonl"
    convs_list = [_conv(f"q{i}", f"a{i}") for i in range(50)] + [_long_convs(3, f"多轮{i}") for i in range(50)]
    _write_sft(p, convs_list)
    m = load_or_build(p, eval_ratio=0.3)
    rep = split_balance_report(m)
    assert rep["kept_total"] == 100 and rep["dedup_removed"] == 0
    assert rep["multi_turn_gap_pp"] <= 5.0  # n=100 时随机分桶的抽样噪声（正式断言只对全量清单生效）


# ---------- SFTDataset 集成（C3/C4/C5 端到端） ----------

def test_sft_dataset_filters_zero_loss_with_remap(tmp_path, tokenizer):
    """C3：训练视图跳过零损失样本并计数；直接索引（__getitem__）不做过滤。"""
    p = tmp_path / "s.jsonl"
    good = _conv("正常问题", "正常回答。")
    oversized = _conv("问", "超长回答。" * 400)  # 硬截 → 零损失
    _write_sft(p, [oversized, good])
    ds = SFTDataset(p, tokenizer, max_length=128, eval_ratio=0.0, add_system_ratio=0.0)
    tv = ds.train_view()
    assert len(tv) == 2  # 视图长度不变（重映射不改位置数）
    item = tv[0]  # 位置 0 原是 oversized → 重映射到 good
    assert bool((item["labels"] != -100).any())
    assert tv.remap_events >= 1
    # 直接索引不做过滤：位置 0 原样返回零损失样本（口径：过滤只发生在训练视图取数路径）
    direct = ds[0]
    assert not bool((direct["labels"] != -100).any())


def test_sft_dataset_dedups_and_splits_by_hash(tmp_path, tokenizer):
    """C4+C5 端到端：重复回复只进一次训练；val 与 train 都来自哈希分桶。"""
    p = tmp_path / "s.jsonl"
    convs_list = [_conv(f"问题{i}", f"回答{i}") for i in range(40)]
    convs_list += [_conv("重复问法", "回答0"), _conv("再换问法", "回答0")]  # 两条重复
    _write_sft(p, convs_list)
    ds = SFTDataset(p, tokenizer, max_length=128, eval_ratio=0.25, add_system_ratio=0.0)
    card = ds.data_card()
    assert card["dedup_removed"] == 2
    assert card["multi_turn_gap_pp"] <= SFTDataset.MULTI_TURN_GAP_TOLERANCE_PP + 1e-9
    tv, ev = ds.train_view(), ds.eval_view()
    assert len(tv) + len(ev) == 40  # 42 行去重 2 → 40
    # 训练视图内回复唯一
    seen: set[int] = set()
    from minimind_reborn.data.manifest import reply_text as rt

    for i in [*tv._indices, *ev._indices]:
        h = stable_hash(normalize_text(rt(json.loads(p.read_text(encoding="utf-8").splitlines()[i])["conversations"])))
        assert h not in seen
        seen.add(h)


def test_sft_ordered_file_eval_not_tail_biased(tmp_path, tokenizer):
    """C5 核心回归：前 80 条单轮 + 后 20 条多轮的有序文件，eval 桶必须两类都有。"""
    p = tmp_path / "ordered.jsonl"
    convs_list = [_conv(f"单轮问题{i}", f"单轮回答{i}") for i in range(80)]
    convs_list += [_long_convs(3, f"多轮{i}") for i in range(20)]
    _write_sft(p, convs_list)
    ds = SFTDataset(p, tokenizer, max_length=256, eval_ratio=0.2, add_system_ratio=0.0)
    ev = ds.eval_view()
    multi_in_eval = sum(1 for i in ev._indices if i >= 80)
    assert 0 < multi_in_eval < len(ev), "哈希分桶下 eval 不应被尾部多轮整段霸占"


def test_manifest_rebuilds_when_file_changes(tmp_path, tokenizer):
    """指纹（mtime+size）失效后自动重建，行数与内容对齐。"""
    import os

    p = tmp_path / "s.jsonl"
    _write_sft(p, [_conv("q", "a")])
    SFTDataset(p, tokenizer, max_length=64, eval_ratio=0.0, add_system_ratio=0.0)
    mpath = manifest_path(p)
    assert mpath.exists()
    with np.load(mpath, allow_pickle=False) as z:
        old_fp = str(z["fingerprint"])
    # 追加一行（mtime 与 size 均变）→ 重建
    p.write_text(
        p.read_text(encoding="utf-8")
        + "\n"
        + json.dumps({"conversations": _conv("q2", "a2")}, ensure_ascii=False),
        encoding="utf-8",
    )
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    ds2 = SFTDataset(p, tokenizer, max_length=64, eval_ratio=0.0, add_system_ratio=0.0)
    assert len(ds2) == 2
    with np.load(mpath, allow_pickle=False) as z:
        assert str(z["fingerprint"]) != old_fp


# ---------- 档二 D1：多轮上采样 ----------

def test_oversample_repeat_counts():
    """单轮 ×1、多轮 ×2、深多轮 ×3（重训计划档二：治多轮占比过低的确定性展开）。"""
    import numpy as np

    from minimind_reborn.data.datasets import (
        DEEP_TURN_REPEAT,
        MULTI_TURN_REPEAT,
        oversample_repeat_counts,
    )
    counts = oversample_repeat_counts(np.array([2, 3, 4, 5, 9]))
    assert counts.tolist() == [1, MULTI_TURN_REPEAT, MULTI_TURN_REPEAT, DEEP_TURN_REPEAT, DEEP_TURN_REPEAT]


def test_sft_dataset_train_view_expands_multi_turn(tmp_path, tokenizer):
    """训练视图按倍数展开、验证视图不展开、data_card 记录有效占比。"""
    p = tmp_path / "s.jsonl"
    single = _conv("单轮问题", "单轮回答")
    multi = [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
        {"role": "user", "content": "第二问"},
        {"role": "assistant", "content": "第二答"},
    ]  # 4 消息：多轮 ×2
    deep = multi + [
        {"role": "user", "content": "第三问"},
        {"role": "assistant", "content": "第三答"},
    ]  # 6 消息：深多轮 ×3
    _write_sft(p, [single, multi, deep])
    ds = SFTDataset(p, tokenizer, max_length=256, eval_ratio=0.0, add_system_ratio=0.0)
    tv = ds.train_view()
    # raw 3 行；有效 = 1 + 2 + 3 = 6
    card = ds.data_card()
    assert card["train_rows_raw"] == 3
    assert card["train_rows_effective"] == 6
    assert card["multi_turn_effective_share"] == round(5 / 6, 4)
    # 展开视图里多轮行索引出现 2 次、深多轮 3 次、单轮 1 次
    from collections import Counter

    c = Counter(tv._indices)
    assert c[0] == 1 and c[1] == 2 and c[2] == 3
    # 直接索引（__getitem__）不受展开影响：仍是原始行
    assert len(ds) == 3


def test_sft_dataset_eval_view_never_expanded(tmp_path, tokenizer):
    p = tmp_path / "s.jsonl"
    convs_list = [
        [
            {"role": "user", "content": f"问{i}"},
            {"role": "assistant", "content": f"答{i}"},
            {"role": "user", "content": f"追问{i}"},
            {"role": "assistant", "content": f"补充{i}"},
        ]
        for i in range(40)
    ]
    _write_sft(p, convs_list)
    ds = SFTDataset(p, tokenizer, max_length=256, eval_ratio=0.25, add_system_ratio=0.0)
    ev = ds.eval_view()
    from collections import Counter

    assert ev is not None
    c = Counter(ev._indices)
    assert max(c.values()) == 1, "验证视图不允许重复样本"

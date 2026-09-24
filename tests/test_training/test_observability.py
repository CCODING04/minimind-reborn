"""可观测性回归测试（2026-09-24 两处真实缺陷的锁定）：

1. metrics step 统一：train/val 同一 optimizer-step 尺度，序列单调不减；
2. ETA 公式：必须用 epoch 开局总耗时，窗口耗时会得到恒 0。
"""

from __future__ import annotations

import json
import os

from minimind_reborn.training.common.trainer import estimate_eta_minutes

from .test_smoke_chain import _build_trainer, _make_cfg


def test_eta_formula():
    """5000 步用了 600 秒 → 剩 3000 步应需 360 秒 = 6 分钟。"""
    assert estimate_eta_minutes(600, 5000, 3000) == 6.0
    assert estimate_eta_minutes(600, 5000, 5000) == 10.0
    assert estimate_eta_minutes(0, 0, 100) == 0.0  # 无已完成步：不外推


def test_metrics_steps_monotonic_and_unified(tmp_path, tiny_jsonl):
    """train/loss 与 val/loss 的 step 必须同尺度且整体单调不减（DDP 双写/双尺子缺陷的回归锁定）。"""
    pretrain_path, _ = tiny_jsonl
    tokenizer = None
    from minimind_reborn.training.common.setup import load_tokenizer

    tokenizer = load_tokenizer()
    cfg = _make_cfg(tmp_path, "pretrain")
    cfg.train.log_interval = 1
    cfg.data.eval_iters = 1
    from minimind_reborn.loggers import setup_logging

    setup_logging(cfg.run_dir)
    trainer = _build_trainer(cfg, tokenizer, pretrain_path, "pretrain")
    trainer.run()

    metrics_file = os.path.join(cfg.run_dir, "metrics.jsonl")
    assert os.path.exists(metrics_file)
    entries = [json.loads(line) for line in open(metrics_file, encoding="utf-8")]
    assert entries, "metrics.jsonl 为空"
    steps = [e["step"] for e in entries]
    assert steps == sorted(steps), f"metrics step 非单调：{steps}"
    train_steps = [e["step"] for e in entries if e["key"].startswith("train/")]
    val_steps = [e["step"] for e in entries if e["key"].startswith("val/")]
    assert val_steps, "缺少 val 序列"
    assert max(val_steps) <= max(train_steps), "val 与 train 不在同一 step 尺度上"

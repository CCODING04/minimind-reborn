"""tiny 模型全链路冒烟（testing-quality §3：数据→训练→保存→续训→生成）。

CPU 快层；GPU 路径由 configs/smoke/*.yaml 在 make smoke 中覆盖。
"""

from __future__ import annotations

import json
import os

import pytest

from minimind_reborn.configuration import load_config
from minimind_reborn.inference.engine import MiniMindLLM
from minimind_reborn.models.weights import load_inference_weights
from minimind_reborn.training.common.setup import build_model, load_tokenizer
from minimind_reborn.training.common.trainer import Trainer


@pytest.fixture()
def tiny_jsonl(tmp_path):
    lines = [{"text": f"这是第{i}条测试文本，用于冒烟训练。" * 3} for i in range(64)]
    p = tmp_path / "pretrain_smoke.jsonl"
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines), encoding="utf-8")
    convs = [
        {
            "conversations": [
                {"role": "user", "content": f"问题{i}"},
                {"role": "assistant", "content": f"回答{i}，这是测试回复。"},
            ]
        }
        for i in range(64)
    ]
    p2 = tmp_path / "sft_smoke.jsonl"
    p2.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in convs), encoding="utf-8")
    return p, p2


def _make_cfg(tmp_path, stage):
    overrides = [
        f"stage={stage}",
        "train.max_steps=2",
        "train.batch_size=2",
        "train.gradient_accumulation_steps=1",
        "train.log_interval=1",
        "train.save_interval_steps=100",
        "train.epochs=1",
        "data.max_seq_len=64",
        "data.eval_ratio=0.125",
        "data.eval_iters=1",
        "data.num_workers=0",
        "model.hidden_size=32",
        "model.num_hidden_layers=2",
        "model.num_attention_heads=2",
        "model.num_key_value_heads=1",
    ]
    cfg = load_config(None, overrides)
    cfg.output_dir = str(tmp_path / "out")
    cfg.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.run_dir = str(tmp_path / "run")
    return cfg


def _build_trainer(cfg, tokenizer, data_path, stage):
    from minimind_reborn.data.datasets import PretrainDataset, SFTDataset

    cls = PretrainDataset if stage == "pretrain" else SFTDataset
    ds = cls(data_path, tokenizer, max_length=cfg.data.max_seq_len, eval_ratio=cfg.data.eval_ratio)

    def compute_loss(batch, m):
        out = m(batch["input_ids"], labels=batch["labels"])
        return out.loss + out.aux_loss, {"train/loss": out.loss.item()}

    return Trainer(
        cfg,
        build_model(cfg, tokenizer),
        tokenizer,
        ds.train_view(),
        ds.eval_view(),
        compute_loss=compute_loss,
        save_weight="smoke",
        device="cpu",
    )


def test_full_chain_train_save_resume_generate(tmp_path, tiny_jsonl):
    pretrain_path, _ = tiny_jsonl
    tokenizer = load_tokenizer()

    # 1) 训练 2 步：max_steps 截断 + 保存 + 四件套落盘
    cfg = _make_cfg(tmp_path, "pretrain")
    from minimind_reborn.loggers import setup_logging

    setup_logging(cfg.run_dir)  # workflow 入口职责；直连 Trainer 的测试里手动触发
    trainer = _build_trainer(cfg, tokenizer, pretrain_path, "pretrain")
    trainer.run()
    weights = os.path.join(cfg.output_dir, f"smoke_{cfg.model.hidden_size}.pth")
    assert os.path.exists(weights), "训练后推理权重未落盘"
    for f in ("config.json", "env.json", "metrics.jsonl", "train.log"):
        assert os.path.exists(os.path.join(cfg.run_dir, f)), f"可追溯四件套缺 {f}"

    # 2) 续训：checkpoint 在场自动恢复（铁律 8）
    cfg2 = _make_cfg(tmp_path, "pretrain")
    cfg2.checkpoint_dir, cfg2.output_dir = cfg.checkpoint_dir, cfg.output_dir
    cfg2.run_dir = str(tmp_path / "run2")
    trainer2 = _build_trainer(cfg2, tokenizer, pretrain_path, "pretrain")
    assert trainer2.start_epoch == 0 and trainer2.start_step == 2, "续训未恢复进度"

    # 3) 引擎（组件注入构造）→ 生成 + request 追溯字段
    model = build_model(cfg2, tokenizer)
    load_inference_weights(model, weights, strict=True)
    llm = MiniMindLLM(model, tokenizer, device="cpu", dtype=model.lm_head.weight.dtype)
    result = llm.generate_text([{"role": "user", "content": "测试"}])
    assert result["content"] is not None and result["finish_reason"] in ("stop", "length")
    assert "request_id" in result and "latency_ms" in result


def test_sft_labels_ignore_non_assistant(tmp_path, tiny_jsonl, tokenizer):
    """SFT 标签语义：非 assistant 区间必须为 -100，回答区间保留 token id。"""
    from minimind_reborn.data.datasets import SFTDataset

    _, sft_path = tiny_jsonl
    ds = SFTDataset(sft_path, tokenizer, max_length=64)
    labels = ds[0]["labels"]
    assert (labels == -100).sum() > 0
    kept = labels[labels != -100]
    assert len(kept) > 0

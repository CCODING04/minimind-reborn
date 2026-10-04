"""SFT workflow：与 pretrain 的唯一差异是数据集（这就是抽基件的意义）。"""

from __future__ import annotations

from minimind_reborn.configuration.schemas import RunConfig
from minimind_reborn.loggers import get_logger, setup_logging
from minimind_reborn.training.common.setup import build_lm_dataset, build_model, load_tokenizer
from minimind_reborn.training.common.trainer import Trainer

logger = get_logger("sft")


def run(cfg: RunConfig, *, device: str | None = None, local_rank: int = 0) -> Trainer:
    setup_logging(cfg.run_dir)
    tokenizer = load_tokenizer()
    model = build_model(cfg, tokenizer)
    train_ds, eval_ds = build_lm_dataset(cfg, tokenizer, kind="sft")
    # 数据卡从底层数据集取（train_ds 是 IndexListView 视图、无 data_card）——
    # 上轮训练 data_card.json 未落盘的根因即此处检查错对象
    sft_base = getattr(train_ds, "_base", None) or train_ds

    def compute_loss(batch, m):
        out = m(batch["input_ids"], labels=batch["labels"])
        loss = out.loss + out.aux_loss
        return loss, {"train/loss": out.loss.item(), "train/aux_loss": float(out.aux_loss)}

    trainer = Trainer(
        cfg,
        model,
        tokenizer,
        train_ds,
        eval_ds,
        compute_loss=compute_loss,
        save_weight="full_sft",
        device=device,
        local_rank=local_rank,
    )
    trainer.run()
    # D5 数据卡：去重/分桶/零损失重映射统计落盘（rank0 写，避免双 rank 竞争）
    if local_rank == 0 and hasattr(sft_base, "data_card"):
        import json
        from pathlib import Path

        card = sft_base.data_card()
        out_path = Path(cfg.run_dir) / "data_card.json"
        out_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("数据卡已写入 %s：去重 %s 条，C3 重映射 %s 次",
                    out_path, card.get("dedup_removed"), card.get("zero_loss_remap_events"))
    return trainer

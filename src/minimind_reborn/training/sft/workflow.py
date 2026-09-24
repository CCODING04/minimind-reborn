"""SFT workflow：与 pretrain 的唯一差异是数据集（这就是抽基件的意义）。"""
from __future__ import annotations

from minimind_reborn.configuration.schemas import RunConfig
from minimind_reborn.loggers import setup_logging
from minimind_reborn.training.common.setup import build_lm_dataset, build_model, load_tokenizer
from minimind_reborn.training.common.trainer import Trainer


def run(cfg: RunConfig, *, device: str | None = None, local_rank: int = 0) -> Trainer:
    setup_logging(cfg.run_dir)
    tokenizer = load_tokenizer()
    model = build_model(cfg, tokenizer)
    train_ds, eval_ds = build_lm_dataset(cfg, tokenizer, kind="sft")

    def compute_loss(batch, m):
        out = m(batch["input_ids"], labels=batch["labels"])
        loss = out.loss + out.aux_loss
        return loss, {"train/loss": out.loss.item(), "train/aux_loss": float(out.aux_loss)}

    trainer = Trainer(
        cfg, model, tokenizer, train_ds, eval_ds,
        compute_loss=compute_loss, save_weight="full_sft", device=device, local_rank=local_rank,
    )
    trainer.run()
    return trainer

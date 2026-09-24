"""预训练 workflow：差异只有数据集与损失——其余全部来自 Trainer 基件。"""
from __future__ import annotations

from minimind_reborn.configuration.schemas import RunConfig
from minimind_reborn.loggers import catch_main, setup_logging
from minimind_reborn.training.common.setup import build_lm_dataset, build_model, load_tokenizer
from minimind_reborn.training.common.trainer import Trainer
from minimind_reborn.utils import dist


def run(cfg: RunConfig, *, device: str | None = None, local_rank: int = 0) -> Trainer:
    setup_logging(cfg.run_dir)
    tokenizer = load_tokenizer()
    model = build_model(cfg, tokenizer)
    train_ds, eval_ds = build_lm_dataset(cfg, tokenizer, kind="pretrain")

    def compute_loss(batch, m):
        out = m(batch["input_ids"], labels=batch["labels"])
        loss = out.loss + out.aux_loss  # aux 仅 MoE 非零；dense 恒 0，加法等价官方
        return loss, {
            "train/loss": out.loss.item(),
            "train/aux_loss": float(out.aux_loss),
        }

    trainer = Trainer(
        cfg, model, tokenizer, train_ds, eval_ds,
        compute_loss=compute_loss, save_weight="pretrain", device=device, local_rank=local_rank,
    )
    trainer.run()
    return trainer

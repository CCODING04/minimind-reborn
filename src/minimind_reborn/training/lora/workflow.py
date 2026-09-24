"""LoRA workflow：冻结基模只训适配器；保存的是 LoRA 权重（经 save_weights_fn 钩子）。"""

from __future__ import annotations

from minimind_reborn.configuration.schemas import RunConfig
from minimind_reborn.loggers import get_logger, setup_logging
from minimind_reborn.models.lora import apply_lora
from minimind_reborn.training.common.setup import build_lm_dataset, build_model, load_tokenizer
from minimind_reborn.training.common.trainer import Trainer
from minimind_reborn.utils.io import atomic_save

logger = get_logger("lora")


def run(cfg: RunConfig, *, device: str | None = None, local_rank: int = 0) -> Trainer:
    setup_logging(cfg.run_dir)
    tokenizer = load_tokenizer()
    model = build_model(cfg, tokenizer)
    apply_lora(model, rank=cfg.lora.rank)

    lora_params = [p for n, p in model.named_parameters() if "lora" in n]
    for n, p in model.named_parameters():
        p.requires_grad = "lora" in n
    total = sum(p.numel() for p in model.parameters())
    lora_count = sum(p.numel() for p in lora_params)
    logger.info(
        "LoRA rank=%d：%.3fM/%.3fM（%.2f%%）", cfg.lora.rank, lora_count / 1e6, total / 1e6, 100 * lora_count / total
    )

    train_ds, eval_ds = build_lm_dataset(cfg, tokenizer, kind="sft")

    def compute_loss(batch, m):
        out = m(batch["input_ids"], labels=batch["labels"])
        loss = out.loss + out.aux_loss
        return loss, {"train/loss": out.loss.item(), "train/aux_loss": float(out.aux_loss)}

    def save_lora_only(m, path):
        raw = m.module if hasattr(m, "module") else m
        raw = getattr(raw, "_orig_mod", raw)
        atomic_save(
            {
                f"{n}.lora.{k}": v.detach().cpu().half()
                for n, mod in raw.named_modules()
                if hasattr(mod, "lora")
                for k, v in mod.lora.state_dict().items()
            },
            path,
        )

    trainer = Trainer(
        cfg,
        model,
        tokenizer,
        train_ds,
        eval_ds,
        compute_loss=compute_loss,
        save_weight=cfg.lora.lora_name,
        device=device,
        local_rank=local_rank,
        save_weights_fn=save_lora_only,
        optimizer_params=lora_params,
    )
    trainer.run()
    return trainer

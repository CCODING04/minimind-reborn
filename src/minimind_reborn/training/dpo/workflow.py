"""DPO workflow：策略模型 + 冻结参考模型，差异在 compute_loss。"""
from __future__ import annotations

import torch

from minimind_reborn.configuration.schemas import RunConfig
from minimind_reborn.loggers import get_logger, setup_logging
from minimind_reborn.models.weights import load_inference_weights, resolve_weight_path
from minimind_reborn.training.common.setup import build_model, build_preference_dataset, load_tokenizer
from minimind_reborn.training.common.trainer import Trainer
from minimind_reborn.training.dpo.loss import dpo_loss, logits_to_log_probs

logger = get_logger("dpo")


def run(cfg: RunConfig, *, device: str | None = None, local_rank: int = 0) -> Trainer:
    setup_logging(cfg.run_dir)
    tokenizer = load_tokenizer()
    model = build_model(cfg, tokenizer)

    # 参考模型：与策略同权重、冻结、只前向——引用漂移的"锚"
    ref_model = build_model(cfg, tokenizer)
    init = resolve_weight_path(cfg.train.init_from or "full_sft", cfg.output_dir, cfg.model.hidden_size, cfg.model.use_moe)
    load_inference_weights(ref_model, init, strict=True)
    ref_model.eval().requires_grad_(False).to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    load_inference_weights(model, init, strict=True)  # 策略模型同起点（Trainer 的 resume 在其后仍可接管）
    logger.info("DPO 起点：%s | beta=%.3f", init.name, cfg.dpo.beta)

    train_ds = build_preference_dataset(cfg, tokenizer)

    def compute_loss(batch, m):
        # chosen/rejected 拼成一个 batch 前后两半，一次前向
        x = torch.cat([batch["x_chosen"], batch["x_rejected"]], dim=0)
        y = torch.cat([batch["y_chosen"], batch["y_rejected"]], dim=0)
        mask = torch.cat([batch["mask_chosen"], batch["mask_rejected"]], dim=0)
        with torch.no_grad():
            ref_logp = logits_to_log_probs(ref_model(x).logits, y)
        out = m(x)
        policy_logp = logits_to_log_probs(out.logits, y)
        loss_val = dpo_loss(ref_logp, policy_logp, mask, beta=cfg.dpo.beta)
        loss = loss_val + out.aux_loss
        return loss, {"train/dpo_loss": loss_val.item(), "train/aux_loss": float(out.aux_loss)}

    trainer = Trainer(
        cfg, model, tokenizer, train_ds, None,
        compute_loss=compute_loss, save_weight="dpo", device=device, local_rank=local_rank,
    )
    trainer.run()
    return trainer

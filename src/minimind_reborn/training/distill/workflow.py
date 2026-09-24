"""蒸馏 workflow：学生 = 当前模型，教师 = 配置指定的更大/同构权重（eval + no_grad）。"""

from __future__ import annotations

import torch

from minimind_reborn.configuration.schemas import RunConfig
from minimind_reborn.loggers import get_logger, setup_logging
from minimind_reborn.models.config import MiniMindConfig
from minimind_reborn.models.model import MiniMindForCausalLM
from minimind_reborn.models.weights import load_inference_weights, resolve_weight_path
from minimind_reborn.training.common.setup import build_lm_dataset, build_model, load_tokenizer
from minimind_reborn.training.common.trainer import Trainer
from minimind_reborn.training.distill.loss import distillation_loss

logger = get_logger("distill")


def run(cfg: RunConfig, *, device: str | None = None, local_rank: int = 0) -> Trainer:
    setup_logging(cfg.run_dir)
    tokenizer = load_tokenizer()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    student = build_model(cfg, tokenizer)

    dc = cfg.distill
    teacher_cfg = MiniMindConfig(
        hidden_size=dc.teacher_hidden_size,
        num_hidden_layers=dc.teacher_num_layers,
        use_moe=dc.teacher_use_moe,
        vocab_size=cfg.model.vocab_size,
        # attention 拓扑跟随 student 配置（teacher 只在 hidden/layers/MoE 上与学生不同）
        num_attention_heads=cfg.model.num_attention_heads,
        num_key_value_heads=cfg.model.num_key_value_heads,
        head_dim=cfg.model.head_dim,
    )
    teacher = MiniMindForCausalLM(teacher_cfg)
    teacher_path = resolve_weight_path(dc.teacher_weight, cfg.output_dir, dc.teacher_hidden_size, dc.teacher_use_moe)
    load_inference_weights(teacher, teacher_path, strict=True)
    teacher.eval().requires_grad_(False).to(device)
    logger.info("教师权重：%s（%.2fM）", teacher_path.name, sum(p.numel() for p in teacher.parameters()) / 1e6)

    train_ds, eval_ds = build_lm_dataset(cfg, tokenizer, kind="sft")
    vocab_student = cfg.model.vocab_size

    def compute_loss(batch, m):
        labels = batch["labels"]
        out = m(batch["input_ids"], labels=labels)
        # 仅对回答区间（label != -100，右移对齐）做 CE 与 KL
        mask = (labels[..., 1:] != -100).float().view(-1)
        shift_labels = labels[..., 1:].contiguous().view(-1)
        logits = out.logits[..., :-1, :].contiguous()
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), shift_labels, ignore_index=-100, reduction="none"
        )
        ce_mean = (ce * mask).sum() / (mask.sum() + 1e-8)
        with torch.no_grad():
            t_logits = teacher(batch["input_ids"]).logits[..., :-1, :][..., :vocab_student].contiguous()
        flat_s = logits.view(-1, logits.size(-1))[mask == 1]
        flat_t = t_logits.view(-1, t_logits.size(-1))[mask == 1]
        kl = distillation_loss(flat_s, flat_t, temperature=dc.temperature)
        loss = (dc.alpha * ce_mean + (1 - dc.alpha) * kl) + out.aux_loss
        return loss, {"train/ce": ce_mean.item(), "train/kl": kl.item(), "train/aux_loss": float(out.aux_loss)}

    trainer = Trainer(
        cfg,
        student,
        tokenizer,
        train_ds,
        eval_ds,
        compute_loss=compute_loss,
        save_weight="full_dist",
        device=device,
        local_rank=local_rank,
    )
    trainer.run()
    return trainer

"""GRPO workflow（官方 train_grpo.py 的架构化重构）。

组相对策略优化：同 prompt 采 num_generations 条回复，组内 z-score 归一化优势，
策略更新带 KL 锚到参考模型；支持 cispo / grpo 两种 per-token 损失。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler

from minimind_reborn.configuration.schemas import RunConfig
from minimind_reborn.data.datasets import RLAIFDataset
from minimind_reborn.data.registry import file_path, resolve_dataset
from minimind_reborn.loggers import get_logger
from minimind_reborn.models.weights import load_inference_weights, resolve_weight_path
from minimind_reborn.training.common.optim import configure_optimizers
from minimind_reborn.training.common.reward_model import LMForRewardModel
from minimind_reborn.training.common.rewards import batch_rewards
from minimind_reborn.training.common.rl_utils import RLSession
from minimind_reborn.training.common.setup import build_model, load_tokenizer
from minimind_reborn.training.rollout import create_rollout_engine
from minimind_reborn.utils import dist
from minimind_reborn.utils.seed import loader_kwargs, set_seed

logger = get_logger("grpo")


def run(cfg: RunConfig, *, device: str | None = None, local_rank: int = 0):
    session = RLSession(cfg, save_weight="grpo", device=device, local_rank=local_rank)
    device = session.device
    tokenizer = load_tokenizer()
    model = build_model(cfg, tokenizer)
    rl = cfg.rl
    t = cfg.train

    init = resolve_weight_path(t.init_from or "full_sft", cfg.output_dir, cfg.model.hidden_size, cfg.model.use_moe)
    load_inference_weights(model, init, strict=True)
    model = model.to(device)  # RL 循环不经 Trainer，设备搬运自己负责

    ref_model = build_model(cfg, tokenizer)
    load_inference_weights(ref_model, init, strict=True)
    ref_model.eval().requires_grad_(False).to(device)

    reward_model = LMForRewardModel(rl.reward_model_path, device=device) if rl.reward_model_path else None

    dataset = RLAIFDataset(
        file_path(resolve_dataset(cfg.data.dataset)),
        tokenizer,
        thinking_ratio=rl.thinking_ratio,
    )
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None
    optimizer = configure_optimizers(model, t.learning_rate, t.weight_decay)
    iters = math.ceil(len(dataset) / t.batch_size / dist.get_world_size())
    total_steps = math.ceil(iters / t.gradient_accumulation_steps) * t.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=t.learning_rate / 10)
    start_epoch, start_step = session.try_resume(model, optimizer, scheduler)

    # DDP 包装在优化器之后；RoPE buffer 各 rank 一致无需广播
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    rollout_engine = create_rollout_engine(cfg, model, tokenizer, device)

    set_seed(t.seed + dist.get_rank(), deterministic=t.deterministic)

    for epoch in range(start_epoch, t.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        loader = DataLoader(dataset, batch_size=t.batch_size, sampler=sampler, **loader_kwargs(cfg.data.num_workers))
        for step, batch in enumerate(loader, start=start_step + 1):
            policy_loss_val, stats = _grpo_step(
                cfg, model, ref_model, rollout_engine, reward_model, tokenizer, batch, optimizer, session
            )
            if step % t.gradient_accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], t.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % t.log_interval == 0:
                session.metrics.log(stats, epoch * iters + step)
                logger.info(
                    "Epoch[%d/%d](%d/%d) %s lr=%.2e",
                    epoch + 1,
                    t.epochs,
                    step,
                    len(loader),
                    " ".join(f"{k}={v:.4f}" for k, v in stats.items()),
                    optimizer.param_groups[0]["lr"],
                )

            if 0 < cfg.train.max_steps <= step:
                logger.info("达到 max_steps=%d 截断（冒烟/调试预算）", cfg.train.max_steps)
                session.save(model, optimizer, scheduler, epoch, step)
                rollout_engine.update_policy(model)
                session.close()
                return model
            if step % t.save_interval_steps == 0 or step == len(loader):
                session.save(model, optimizer, scheduler, epoch, step, extra_states={"scheduler": scheduler})
                rollout_engine.update_policy(model)
        start_step = 0

    session.close()
    return model


def _grpo_step(cfg, model, ref_model, rollout_engine, reward_model, tokenizer, batch, optimizer, session):
    rl = cfg.rl
    device = session.device
    prompts = batch["prompt"]
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False,
        padding_side="left",
        add_special_tokens=False,
    ).to(device)
    if cfg.data.max_seq_len:
        enc["input_ids"] = enc["input_ids"][:, -cfg.data.max_seq_len :]
        enc["attention_mask"] = enc["attention_mask"][:, -cfg.data.max_seq_len :]

    result = rollout_engine.rollout(
        prompt_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        num_generations=rl.num_generations,
        max_new_tokens=rl.max_gen_len,
        temperature=0.8,
    )
    outputs = result.output_ids
    completion_ids = result.completion_ids
    old_per_token_logps = result.per_token_logps.to(device).detach()
    prompt_lens = result.prompt_lens.to(device)
    completions = result.completions

    # 奖励：规则 + 可选 RM；group 内归一化优势
    flat_prompts = [p for p in prompts for _ in range(rl.num_generations)]
    rewards = batch_rewards(flat_prompts, completions, reward_model, device)
    grouped = rewards.view(-1, rl.num_generations)
    mean_r = grouped.mean(dim=1).repeat_interleave(rl.num_generations)
    std_r = grouped.std(dim=1, unbiased=False).repeat_interleave(rl.num_generations)
    advantages = (rewards - mean_r) / (std_r + 1e-4)

    # 生成位掩码：截到首个 eos（含）
    completion_pad_mask = result.completion_mask.to(device).bool()
    is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask
    eos_idx = torch.full((is_eos.shape[0],), is_eos.shape[1] - 1, dtype=torch.long, device=device)
    hit = is_eos.any(dim=1)
    eos_idx[hit] = is_eos.int().argmax(dim=1)[hit]
    ar = torch.arange(is_eos.shape[1], device=device).expand(is_eos.shape[0], -1)
    completion_mask = ((ar <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()

    # 策略 / 参考的逐 token logp（只取生成位）
    with session.autocast_ctx:
        res = model(outputs, attention_mask=(outputs != tokenizer.pad_token_id).long())
        aux_loss = res.aux_loss if cfg.model.use_moe else torch.tensor(0.0, device=device)
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.shape[1], device=device).unsqueeze(0)
        per_token_logps = (
            F.log_softmax(res.logits[:, :-1, :], dim=-1)
            .gather(2, outputs[:, 1:].unsqueeze(-1))
            .squeeze(-1)
            .gather(1, logp_pos)
        )
    with torch.no_grad():
        ref_logps = (
            F.log_softmax(
                ref_model(outputs, attention_mask=(outputs != tokenizer.pad_token_id).long()).logits[:, :-1, :], dim=-1
            )
            .gather(2, outputs[:, 1:].unsqueeze(-1))
            .squeeze(-1)
            .gather(1, logp_pos)
        )

    # per-token 损失：cispo（钳制比率系数）或 grpo（双面 clip）+ k3 KL 惩罚
    kl_div = ref_logps - per_token_logps
    per_token_kl = torch.exp(kl_div) - kl_div - 1
    ratio = torch.exp(per_token_logps - old_per_token_logps)
    if rl.loss_type == "cispo":
        clamped = torch.clamp(ratio, max=rl.epsilon_high).detach()
        per_token_loss = -(clamped * advantages.unsqueeze(1) * per_token_logps - rl.beta * per_token_kl)
    else:
        clipped = torch.clamp(ratio, 1 - rl.epsilon, 1 + rl.epsilon)
        per_token_loss = -(
            torch.min(ratio * advantages.unsqueeze(1), clipped * advantages.unsqueeze(1)) - rl.beta * per_token_kl
        )
    mask = completion_mask.float()
    policy_loss = ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
    loss = (policy_loss + aux_loss) / cfg.train.gradient_accumulation_steps
    loss.backward()

    stats = {
        "train/reward": rewards.mean().item(),
        "train/kl_ref": ((ref_logps - per_token_logps) * mask).sum().item() / max(mask.sum().item(), 1),
        "train/adv_std": advantages.std().item(),
        "train/policy_loss": policy_loss.item(),
        "train/avg_response_len": mask.sum(dim=1).float().mean().item(),
    }
    return policy_loss.item(), stats

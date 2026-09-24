"""PPO workflow（官方 train_ppo.py 的架构化重构）。

actor + critic + 冻结 ref + 可选 RM + rollout 引擎；
GAE 优势估计、双面 clip 策略损失、clip 值损失、KL 早停（跨 rank all_reduce 防 DDP 死锁）。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler

from minimind_reborn.configuration.schemas import RunConfig
from minimind_reborn.data.datasets import RLAIFDataset
from minimind_reborn.data.registry import file_path, resolve_dataset
from minimind_reborn.loggers import get_logger
from minimind_reborn.models.weights import load_inference_weights, resolve_weight_path
from minimind_reborn.training.common.optim import configure_optimizers
from minimind_reborn.training.common.rewards import batch_rewards
from minimind_reborn.training.common.reward_model import LMForRewardModel
from minimind_reborn.training.common.rl_utils import RLSession
from minimind_reborn.training.common.setup import build_model, load_tokenizer
from minimind_reborn.training.ppo.critic import CriticModel
from minimind_reborn.training.rollout import create_rollout_engine
from minimind_reborn.utils import dist
from minimind_reborn.utils.seed import loader_kwargs, set_seed

logger = get_logger("ppo")


def run(cfg: RunConfig, *, device: str | None = None, local_rank: int = 0):
    session = RLSession(cfg, save_weight="ppo_actor", device=device, local_rank=local_rank)
    device = session.device
    tokenizer = load_tokenizer()
    rl = cfg.rl
    t = cfg.train

    init = resolve_weight_path(t.init_from or "full_sft", cfg.output_dir, cfg.model.hidden_size, cfg.model.use_moe)
    actor = build_model(cfg, tokenizer)
    load_inference_weights(actor, init, strict=True)
    ref_model = build_model(cfg, tokenizer)
    load_inference_weights(ref_model, init, strict=True)
    ref_model.eval().requires_grad_(False).to(device)
    critic = CriticModel(actor.config)
    load_inference_weights(critic, init, strict=False)  # value_head 随机初始化，其余对齐 actor
    critic = critic.to(device)
    reward_model = LMForRewardModel(rl.reward_model_path, device=device) if rl.reward_model_path else None

    dataset = RLAIFDataset(file_path(resolve_dataset(cfg.data.dataset)), tokenizer, thinking_ratio=rl.thinking_ratio)
    sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
    actor_optimizer = configure_optimizers(actor, t.learning_rate, t.weight_decay)
    critic_optimizer = configure_optimizers(critic, rl.critic_learning_rate, t.weight_decay)
    iters = math.ceil(len(dataset) / t.batch_size / dist.get_world_size())
    mb_factor = max(1, math.ceil(t.batch_size / rl.mini_batch_size))
    total_steps = math.ceil(iters * t.epochs * rl.ppo_update_iters * mb_factor / t.gradient_accumulation_steps)
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=max(total_steps, 1), eta_min=t.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=max(total_steps, 1), eta_min=rl.critic_learning_rate / 10)
    start_epoch, start_step = session.try_resume(
        actor, actor_optimizer, actor_scheduler,
        extra={"critic_model": critic, "critic_optimizer": critic_optimizer, "critic_scheduler": critic_scheduler},
    )

    if dist.is_initialized():
        actor = torch.nn.parallel.DistributedDataParallel(actor, device_ids=[local_rank], broadcast_buffers=False)
        critic = torch.nn.parallel.DistributedDataParallel(critic, device_ids=[local_rank], broadcast_buffers=False)
    rollout_engine = create_rollout_engine(cfg, actor, tokenizer, device)
    set_seed(t.seed + dist.get_rank(), deterministic=t.deterministic)

    for epoch in range(start_epoch, t.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        loader = DataLoader(dataset, batch_size=t.batch_size, sampler=sampler, **loader_kwargs(t.num_workers))
        for step, batch in enumerate(loader, start=start_step + 1):
            stats = _ppo_step(cfg, session, actor, critic, ref_model, rollout_engine, reward_model,
                              tokenizer, batch, actor_optimizer, critic_optimizer,
                              actor_scheduler, critic_scheduler)
            if step % t.log_interval == 0:
                session.metrics.log(stats, epoch * iters + step)
                logger.info("Epoch[%d/%d](%d/%d) %s", epoch + 1, t.epochs, step, len(loader),
                            " ".join(f"{k}={v:.4f}" for k, v in stats.items()))
            if step % t.save_interval_steps == 0 or step == len(loader):
                session.save(actor, actor_optimizer, actor_scheduler, epoch, step,
                             extra_states={"critic_model": critic, "critic_optimizer": critic_optimizer,
                                           "critic_scheduler": critic_scheduler})
                rollout_engine.update_policy(actor)
        start_step = 0

    session.close()
    return actor


def _ppo_step(cfg, session, actor, critic, ref_model, rollout_engine, reward_model,
              tokenizer, batch, actor_optimizer, critic_optimizer,
              actor_scheduler, critic_scheduler) -> dict[str, float]:
    rl = cfg.rl
    device = session.device
    prompts = batch["prompt"]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=cfg.data.max_seq_len, padding_side="left").to(device)

    result = rollout_engine.rollout(enc["input_ids"], enc["attention_mask"], num_generations=1,
                                    max_new_tokens=rl.max_gen_len, temperature=0.8)
    gen_out = result.output_ids
    completion_ids = result.completion_ids
    prompt_lens = result.prompt_lens.to(device)
    old_resp_logp = result.per_token_logps.to(device)
    rewards = batch_rewards(prompts, result.completions, reward_model, device)

    B = len(prompts)
    full_mask = (gen_out != tokenizer.pad_token_id).long()
    labels = gen_out[:, 1:].clone()
    resp_idx = torch.arange(completion_ids.shape[1], device=gen_out.device).unsqueeze(0)
    logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
    resp_pad_mask = result.completion_mask.to(device).bool()
    # 生成位长度截到 eos（含）
    resp_lengths = resp_pad_mask.sum(dim=1)
    eos_mask = completion_ids.eq(tokenizer.eos_token_id) & resp_pad_mask
    has_eos = eos_mask.any(dim=1)
    eos_pos = torch.argmax(eos_mask.int(), dim=1)
    resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
    resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()

    with torch.no_grad():
        values_seq = critic(input_ids=gen_out, attention_mask=full_mask)
        old_resp_values = values_seq.gather(1, logp_pos) * resp_policy_mask
        ref_resp_logp = (
            F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1)
            .gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
        )
        # 末位挂外部奖励 → GAE 反序累积
        token_rewards = torch.zeros_like(old_resp_logp)
        last_idx = resp_lengths - 1
        valid = resp_lengths > 0
        token_rewards[torch.arange(B, device=device)[valid], last_idx[valid]] += rewards[valid]
        gen_len = old_resp_values.shape[1]
        lastgaelam = torch.zeros(B, device=device)
        advs_rev = []
        for tpos in reversed(range(gen_len)):
            nv = old_resp_values[:, tpos + 1] if tpos < gen_len - 1 else 0.0
            delta = token_rewards[:, tpos] + rl.gamma * nv - old_resp_values[:, tpos]
            lastgaelam = delta + rl.gamma * rl.lam * lastgaelam
            advs_rev.append(lastgaelam)
        advantages = torch.stack(advs_rev[::-1], dim=1)
        returns = advantages + old_resp_values
        adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
        adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
        advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

    # PPO 内循环：minibatch 多次更新；KL 早停时仍走 0 权重 backward 保持 DDP 通信闭环
    accum = cfg.train.gradient_accumulation_steps
    totals = {"policy": 0.0, "value": 0.0, "kl": 0.0, "clipfrac": 0.0, "n": 0}
    stop_ppo = False
    grad_step = 0
    for _ppo_epoch in range(rl.ppo_update_iters):
        if stop_ppo:
            break
        for inds in torch.randperm(B, device=device).split(rl.mini_batch_size):
            mb_values = critic(input_ids=gen_out[inds], attention_mask=full_mask[inds]).gather(1, logp_pos[inds])
            with session.autocast_ctx:
                res = actor(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                aux_loss = res.aux_loss if cfg.model.use_moe else torch.tensor(0.0, device=device)
                mb_logp = (
                    F.log_softmax(res.logits[:, :-1], dim=-1)
                    .gather(2, labels[inds].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos[inds])
                )
            log_ratio = mb_logp - old_resp_logp[inds]
            approx_kl = (0.5 * (log_ratio**2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
            kl_val = approx_kl.detach().clone()
            if dist.is_initialized():
                dist.all_reduce(kl_val, op=dist.ReduceOp.AVG)  # 防"某卡 break 其余继续"死锁
            if kl_val > rl.early_stop_kl:
                stop_ppo = True
            ratio = torch.exp(log_ratio)
            clipfrac = ((((ratio - 1.0).abs() > rl.clip_epsilon).float() * resp_policy_mask[inds]).sum()
                        / resp_policy_mask[inds].sum().clamp(min=1))
            kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_logp) - (ref_resp_logp[inds] - mb_logp) - 1.0)
                              * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
            policy_loss = ((torch.max(-advantages[inds] * ratio,
                                      -advantages[inds] * torch.clamp(ratio, 1 - rl.clip_epsilon, 1 + rl.clip_epsilon))
                            * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                           + rl.kl_coef * kl_ref_penalty)
            value_loss = 0.5 * (torch.max((mb_values - returns[inds]) ** 2,
                                          (torch.clamp(mb_values, old_resp_values[inds] - rl.cliprange_value,
                                                       old_resp_values[inds] + rl.cliprange_value) - returns[inds]) ** 2)
                                * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
            scale = 0.0 if stop_ppo else 1.0 / accum
            loss = (policy_loss + rl.vf_coef * value_loss + aux_loss) * scale
            loss.backward()
            totals["policy"] += policy_loss.item()
            totals["value"] += value_loss.item()
            totals["kl"] += float(kl_val)
            totals["clipfrac"] += clipfrac.item()
            totals["n"] += 1
            grad_step += 1
            if grad_step % accum == 0:
                clip_grad_norm_(actor.parameters(), cfg.train.grad_clip)
                clip_grad_norm_(critic.parameters(), cfg.train.grad_clip)
                actor_optimizer.step()
                critic_optimizer.step()
                actor_scheduler.step()
                critic_scheduler.step()
                actor_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)

    n = max(totals["n"], 1)
    return {
        "train/reward": rewards.mean().item(),
        "train/policy_loss": totals["policy"] / n,
        "train/critic_loss": totals["value"] / n,
        "train/approx_kl": totals["kl"] / n,
        "train/clipfrac": totals["clipfrac"] / n,
        "train/avg_response_len": resp_lengths.float().mean().item(),
    }

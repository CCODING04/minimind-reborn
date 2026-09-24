"""Agent RL workflow（官方 train_agent.py 的架构化重构）：多轮工具调用 + 组相对策略优化。"""
from __future__ import annotations

import json
import math
import random
import re

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler

from minimind_reborn.configuration.schemas import RunConfig
from minimind_reborn.data.datasets import AgentRLDataset
from minimind_reborn.data.registry import file_path, resolve_dataset
from minimind_reborn.loggers import get_logger
from minimind_reborn.models.weights import load_inference_weights, resolve_weight_path
from minimind_reborn.training.common.optim import configure_optimizers
from minimind_reborn.training.common.rewards import rep_penalty
from minimind_reborn.training.common.rl_utils import RLSession
from minimind_reborn.training.agent.tools import CHECK_ARGS, TOOLS, execute_tool, parse_tool_calls
from minimind_reborn.training.common.setup import build_model, load_tokenizer
from minimind_reborn.training.rollout import create_rollout_engine
from minimind_reborn.utils import dist
from minimind_reborn.utils.seed import loader_kwargs, set_seed

logger = get_logger("agent")


def _validate_gt_in_text(text: str, gt_list) -> set:
    text_num = str(text).replace(",", "")
    nums = [float(x) for x in
            re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])", text_num)]
    found = set()
    for g in gt_list:
        s = str(g).strip()
        if s and s.lower() in str(text).lower():
            found.add(g)
        elif s.replace(",", "").replace(".", "", 1).isdigit() or _is_number(s.replace(",", "")):
            try:
                gv = float(s.replace(",", ""))
                if any(abs(gv - n) < 1e-6 for n in nums):
                    found.add(g)
            except ValueError:
                pass
    return found


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _rollout_single(rollout_engine, tokenizer, messages, tools, *, max_turns, max_new_tokens,
                    thinking_ratio, device):
    """多轮采样：工具观测经模板 marker 对齐回 token 流（观测位 mask=0 不计梯度）。"""
    all_outputs = []
    prompt_ids = None
    response_ids: list[int] = []
    response_mask: list[int] = []
    response_old_logps: list[float] = []
    final_context = ""
    unfinished = False
    open_thinking = random.random() < thinking_ratio
    for turn in range(max_turns):
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                tools=tools, open_thinking=open_thinking)
        if prompt_ids is None:
            prompt_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([prompt_ids + response_ids], device=device)
        result = rollout_engine.rollout(prompt_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                        num_generations=1, max_new_tokens=max_new_tokens, temperature=0.8)
        valid_len = int(result.completion_mask[0].sum().item())
        new_ids = result.completion_ids[0, :valid_len].tolist()
        new_logps = result.per_token_logps[0, :valid_len].tolist()
        new_text = result.completions[0]
        all_outputs.append(new_text)
        response_ids.extend(new_ids)
        response_mask.extend([int(t != tokenizer.eos_token_id) for t in new_ids])
        response_old_logps.extend(new_logps)
        final_context = context + new_text
        calls = parse_tool_calls(new_text)
        if not calls:
            break
        unfinished = turn == max_turns - 1
        assistant_message = {"role": "assistant", "content": new_text}
        messages.append(assistant_message)
        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:  # noqa: BLE001
                    raw = {}
            result_obj = execute_tool(name, raw)
            result_str = (json.dumps(result_obj, ensure_ascii=False)
                          if result_obj else '{"error": "tool not found"}')[:2048]
            messages.append({"role": "tool", "content": result_str})
        # 用 marker 找到观测段在模板中的增量 token（官方同款技巧）
        marker = f"<|agent_observation_{id(messages)}_{len(response_ids)}|>"
        assistant_message["content"] += marker
        marked = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished,
                                               tools=tools, open_thinking=open_thinking)
        assistant_message["content"] = new_text
        _, found, observation = marked.partition(marker)
        if not found:
            raise RuntimeError("chat template 未能保留 assistant 内容边界（模板变更需同步 loss 对齐逻辑）")
        obs_delta = tokenizer(observation, add_special_tokens=False)["input_ids"]
        if new_ids and new_ids[-1] == tokenizer.eos_token_id and obs_delta[:1] == [tokenizer.eos_token_id]:
            obs_delta = obs_delta[1:]
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished,
                                                      tools=tools, open_thinking=open_thinking)
    return (all_outputs[-1] if all_outputs else "", final_context, prompt_ids or [],
            response_ids, response_mask, response_old_logps, list(all_outputs), unfinished)


def _rewards(prompts, completions, gt_batch, tools_batch, num_gen, device, turn_outputs_batch, unfinished_batch):
    """工具对齐分 + GT 命中分 + 格式分（官方语义保留）。"""
    rewards = torch.zeros(len(completions), device=device)
    for idx, response in enumerate(completions):
        reward, answer = 0.0, response
        sample_idx = idx // num_gen
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch is not None else False
        turn_answers = [t.split("</think>", 1)[-1].strip() if "</think>" in t else t.strip() for t in turn_outputs]
        answer = turn_answers[-1] if turn_answers else response.strip()
        valid_names = {t["function"]["name"] for t in tools} if tools else set()
        tool_calls = []
        for turn_answer in turn_answers:
            tool_calls.extend(parse_tool_calls(turn_answer))
        reward -= 0.5 * sum(abs(t.count("<tool_call>") - t.count("</tool_call>")) for t in turn_answers)
        if not tool_calls:
            reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5
            if "</think>" in response:
                think, answer = response.split("</think>", 1)
                reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5
                reward += 0.25 if response.count("</think>") == 1 else -0.25
                answer = answer.strip()
            reward -= rep_penalty(answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)
        else:
            gt = gt_batch[sample_idx]
            valid_count = 0
            for call in tool_calls:
                name, raw = call.get("name", ""), call.get("arguments", {})
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except Exception:  # noqa: BLE001
                        raw = {}
                check = CHECK_ARGS.get(name)
                valid_count += int(bool(name in valid_names and check and check(raw)))
            tool_gap = abs(valid_count - len(gt)) + max(0, len(tool_calls) - valid_count)
            reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap
            final_text = "" if unfinished else (answer.split("</tool_call>")[-1] if "</tool_call>" in answer else answer)
            verified = _validate_gt_in_text(final_text, gt) if gt else set()
            if gt:
                reward += 2.5 * len(verified) / len(gt)
            if unfinished:
                reward -= 0.5
            reward -= rep_penalty(final_text if final_text else answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)
    return rewards


def run(cfg: RunConfig, *, device: str | None = None, local_rank: int = 0):
    session = RLSession(cfg, save_weight="agent", device=device, local_rank=local_rank)
    device = session.device
    tokenizer = load_tokenizer()
    rl = cfg.rl
    t = cfg.train

    model = build_model(cfg, tokenizer)
    init = resolve_weight_path(t.init_from or "full_sft", cfg.output_dir, cfg.model.hidden_size, cfg.model.use_moe)
    load_inference_weights(model, init, strict=True)
    ref_model = build_model(cfg, tokenizer)
    load_inference_weights(ref_model, init, strict=True)
    ref_model.eval().requires_grad_(False).to(device)

    dataset = AgentRLDataset(file_path(resolve_dataset(cfg.data.dataset)), tokenizer)
    collate = lambda b: {"messages": [x["messages"] for x in b], "tools": [x["tools"] for x in b], "gt": [x["gt"] for x in b]}
    sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
    optimizer = configure_optimizers(model, t.learning_rate, t.weight_decay)
    iters = math.ceil(len(dataset) / t.batch_size / dist.get_world_size())
    total_steps = math.ceil(iters / t.gradient_accumulation_steps) * t.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=t.learning_rate / 10)
    start_epoch, start_step = session.try_resume(model, optimizer, scheduler)

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    rollout_engine = create_rollout_engine(cfg, model, tokenizer, device)
    set_seed(t.seed + dist.get_rank(), deterministic=t.deterministic)

    for epoch in range(start_epoch, t.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        loader = DataLoader(dataset, batch_size=t.batch_size, sampler=sampler,
                            collate_fn=collate, **loader_kwargs(t.num_workers))
        for step, batch in enumerate(loader, start=start_step + 1):
            stats, loss_val = _agent_step(cfg, session, model, ref_model, rollout_engine, tokenizer,
                                          batch, optimizer, TOOLS)
            if step % t.gradient_accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], t.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % t.log_interval == 0:
                session.metrics.log(stats, epoch * iters + step)
                logger.info("Epoch[%d/%d](%d/%d) %s", epoch + 1, t.epochs, step, len(loader),
                            " ".join(f"{k}={v:.4f}" for k, v in stats.items()))
            if step % t.save_interval_steps == 0 or step == len(loader):
                session.save(model, optimizer, scheduler, epoch, step, extra_states={"scheduler": scheduler})
                rollout_engine.update_policy(model)
        start_step = 0

    session.close()
    return model


def _agent_step(cfg, session, model, ref_model, rollout_engine, tokenizer, batch, optimizer, tools_schema):
    rl = cfg.rl
    device = session.device
    messages_batch, tools_batch, gt_batch = batch["messages"], batch["tools"], batch["gt"]
    import json as _json
    with torch.no_grad():
        packed = []
        completions, contexts, turn_outputs_batch, unfinished_batch = [], [], [], []
        for messages, tools in zip(messages_batch, tools_batch):
            for _ in range(rl.num_generations):
                msgs = [dict(m) for m in messages]
                completion, context, prompt_ids, resp_ids, resp_mask, old_logps, turns, unfinished = _rollout_single(
                    rollout_engine, tokenizer, msgs, tools, max_turns=rl.max_turns,
                    max_new_tokens=rl.max_gen_len, thinking_ratio=rl.thinking_ratio, device=device)
                completions.append(completion)
                contexts.append(context)
                turn_outputs_batch.append(turns)
                unfinished_batch.append(unfinished)
                mask = [0] * len(prompt_ids) + resp_mask
                ids = prompt_ids + resp_ids
                olds = [0.0] * max(len(prompt_ids) - 1, 0) + old_logps
                if len(ids) > rl.max_total_len:
                    ids, mask, olds = ids[-rl.max_total_len:], mask[-rl.max_total_len:], olds[-(len(ids) - 1):]
                prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
                packed.append((ids, mask, prompt_len, olds))
        seq_lens = torch.tensor([len(x[0]) for x in packed], device=device)
        max_len = int(seq_lens.max().item())
        input_ids = torch.tensor([x[0] + [tokenizer.pad_token_id] * (max_len - len(x[0])) for x in packed], device=device)
        prompt_lens = torch.tensor([x[2] for x in packed], device=device)
        full_resp_masks = torch.tensor([x[1] + [0] * (max_len - len(x[1])) for x in packed], device=device, dtype=torch.float32)
        old_logps = torch.tensor([x[3] + [0.0] * ((max_len - 1) - len(x[3])) for x in packed], device=device, dtype=torch.float32)
        full_mask = (torch.arange(max_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)).long()

    prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t)
               for m, t in zip(messages_batch, tools_batch)]
    rewards = _rewards(prompts, completions, gt_batch, tools_batch, rl.num_generations, device,
                       turn_outputs_batch, unfinished_batch)

    with session.autocast_ctx:
        res = model(input_ids, attention_mask=full_mask)
        aux_loss = res.aux_loss
        logits = res.logits[:, :-1, :]
        per_token_logps = F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        from minimind_reborn.training.rollout import compute_per_token_logps

        ref_per_token = compute_per_token_logps(ref_model, input_ids, input_ids.shape[1] - 1, attention_mask=full_mask)

    completion_mask = full_resp_masks[:, 1:]
    is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
    eos_idx = torch.full((completion_mask.shape[0],), completion_mask.shape[1] - 1, device=device, dtype=torch.long)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    pos = torch.arange(completion_mask.shape[1], device=device).unsqueeze(0)
    completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
    token_counts = completion_mask.sum(dim=1)
    valid_rows = token_counts > 0

    grouped = rewards.view(-1, rl.num_generations)
    mean_r = grouped.mean(dim=1).repeat_interleave(rl.num_generations)
    std_r = grouped.std(dim=1, unbiased=False).repeat_interleave(rl.num_generations)
    advantages = (rewards - mean_r) / (std_r + 1e-4)

    kl_div = ref_per_token - per_token_logps
    per_token_kl = torch.exp(kl_div) - kl_div - 1
    ratio = torch.exp(per_token_logps - old_logps)
    if rl.loss_type == "cispo":
        clamped = torch.clamp(ratio, max=rl.epsilon_high).detach()
        per_token_loss = -(clamped * advantages.unsqueeze(1) * per_token_logps - rl.beta * per_token_kl)
    else:
        clipped = torch.clamp(ratio, 1 - rl.epsilon, 1 + rl.epsilon)
        per_token_loss = -(torch.min(ratio * advantages.unsqueeze(1), clipped * advantages.unsqueeze(1))
                           - rl.beta * per_token_kl)
    if valid_rows.any():
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
    else:
        policy_loss = per_token_loss.sum() * 0.0
    loss = (policy_loss + aux_loss) / cfg.train.gradient_accumulation_steps
    loss.backward()

    stats = {
        "train/reward": rewards.mean().item(),
        "train/kl_ref": ((ref_per_token - per_token_logps) * completion_mask).sum().item() / max(token_counts.sum().item(), 1),
        "train/policy_loss": policy_loss.item(),
        "train/avg_response_len": token_counts.float().mean().item(),
        "train/adv_std": advantages.std().item(),
    }
    return stats, policy_loss.item()

"""Rollout 引擎抽象（官方 rollout_engine.py 重构）：策略采样与训练解耦。

协议而非强制继承（project-structure §4）：同签名 rollout()/update_policy() 即可互换；
torch 原生引擎用于单卡/小模型，sglang HTTP 引擎用于加速大规模 RL（可选）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from torch import Tensor

from minimind_reborn.configuration.schemas import GenerateConfig, RunConfig
from minimind_reborn.inference.generator import generate
from minimind_reborn.loggers import get_logger
from minimind_reborn.models.model import MiniMindForCausalLM
from minimind_reborn.models.weights import unwrap_model

logger = get_logger("rollout")


@dataclass
class RolloutResult:
    output_ids: Tensor  # (b, prompt+completion) 右 padding
    completion_ids: Tensor  # (b, R)
    per_token_logps: Tensor  # (b, R) 采样时策略的逐 token logp
    completions: list[str]  # 解码文本
    prompt_lens: Tensor  # (b,)
    completion_mask: Tensor  # (b, R) 1=有效生成位


def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Tensor | None = None) -> Tensor:
    """给定完整序列，取最后 n_keep 个 token 的策略 logp（教学/对齐检查用）。"""
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    unwrapped = unwrap_model(model)
    # inference tensor 防御：进入 autograd 相关路径前落回普通张量
    ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    logits = unwrapped(ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token = []
    for row_logits, row_ids in zip(logits, ids[:, -n_keep:], strict=False):
        row_ids = row_ids.detach().clone() if row_ids.is_inference() else row_ids
        per_token.append(torch.gather(row_logits.log_softmax(dim=-1), 1, row_ids.unsqueeze(1)).squeeze(1))
    return torch.stack(per_token)


class RolloutEngine(Protocol):
    def rollout(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        num_generations: int,
        max_new_tokens: int,
        temperature: float = 0.8,
    ) -> RolloutResult: ...

    def update_policy(self, model) -> None: ...


class TorchRolloutEngine:
    """torch 原生引擎：走 inference/generator 的预分配 cache 内核。"""

    def __init__(self, policy_model, tokenizer, device: str = "cuda"):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def rollout(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        num_generations: int,
        max_new_tokens: int,
        temperature: float = 0.8,
    ) -> RolloutResult:
        model = cast(MiniMindForCausalLM, unwrap_model(self.policy_model))
        gen = GenerateConfig(temperature=temperature, top_p=0.95, top_k=50)
        # repeat_interleave 到 (b·num_gen, P)；generate 内部预分配 KV cache
        result = generate(
            model,
            prompt_ids.detach().repeat_interleave(num_generations, dim=0),
            gen,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask.detach().repeat_interleave(num_generations, dim=0),
        )
        output_ids = result["sequences"].detach().clone()  # 落回普通张量（inference 防御）
        prompt_len = prompt_ids.shape[1]
        completion_ids = output_ids[:, prompt_len:]
        per_token_logps = compute_per_token_logps(
            self.policy_model,
            output_ids,
            completion_ids.shape[1],
            attention_mask=torch.cat(
                [
                    attention_mask.detach().repeat_interleave(num_generations, dim=0),
                    attention_mask.new_ones(output_ids.shape[0], completion_ids.shape[1]),
                ],
                dim=1,
            ),
        )
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(
            output_ids=output_ids,
            completion_ids=completion_ids,
            per_token_logps=per_token_logps,
            completions=completions,
            prompt_lens=prompt_ids.new_full((output_ids.shape[0],), prompt_len),
            completion_mask=attention_mask.new_ones(output_ids.shape[0], completion_ids.shape[1]),
        )

    def update_policy(self, model) -> None:
        self.policy_model = model


class SGLangRolloutEngine:
    """SGLang HTTP 引擎（可选加速）：权重经磁盘热更新同步。"""

    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "sglang_ckpt", timeout: int = 120):
        import requests
        from transformers import AutoTokenizer

        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self.shared_ckpt_path = shared_ckpt_path
        self.model_path = model_path
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def rollout(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        num_generations: int,
        max_new_tokens: int,
        temperature: float = 0.8,
    ) -> RolloutResult:
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask, strict=False):
            input_ids_list.append(ids[mask.bool()].tolist())
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]
        payload: dict[str, Any] = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,
        }
        resp = self._requests.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        results = resp.json()
        if not isinstance(results, list):
            results = [results]
        outs, comps, comps_ids, logps = [], [], [], []
        for i, item in enumerate(results):
            meta = item.get("meta_info", {})
            c_ids = meta.get("output_ids", item.get("output_ids", []))
            raw_lp = [x[0] if isinstance(x, (list, tuple)) else x for x in meta.get("output_token_logprobs", [])]
            if len(raw_lp) < len(c_ids):
                raw_lp = [0.0] * (len(c_ids) - len(raw_lp)) + raw_lp
            full = all_input_ids[i] + list(c_ids)
            outs.append(full)
            comps_ids.append(list(c_ids))
            logps.append(raw_lp[-len(c_ids) :] if c_ids else [])
            comps.append(self.tokenizer.decode(c_ids, skip_special_tokens=True))
        device = prompt_ids.device
        max_c = max(1, max(len(x) for x in comps_ids))
        max_o = max(len(x) for x in outs) + max_c
        pad_id = self.tokenizer.pad_token_id or 0

        def pad(seqs, width, val):
            return torch.tensor([s + [val] * (width - len(s)) for s in seqs], device=device)

        return RolloutResult(
            output_ids=pad(outs, max_o, pad_id),
            completion_ids=pad(comps_ids, max_c, pad_id),
            per_token_logps=pad(logps, max_c, 0.0).float(),
            completions=cast("list[str]", comps),
            prompt_lens=torch.tensor([len(x) for x in all_input_ids], device=device),
            completion_mask=torch.tensor([[1] * len(x) + [0] * (max_c - len(x)) for x in comps_ids], device=device),
        )

    def update_policy(self, model) -> None:
        """rank0 写盘热更新 → 广播结果（等主进程语义）。"""
        import os

        ok = True
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        ):
            try:
                unwrapped = cast(MiniMindForCausalLM, unwrap_model(model))
                path = os.path.abspath(self.shared_ckpt_path)
                state = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
                unwrapped.save_pretrained(path, state_dict=state, safe_serialization=False)
                self.tokenizer.save_pretrained(path)
                resp = self._requests.post(
                    f"{self.base_url}/update_weights_from_disk", json={"model_path": path}, timeout=self.timeout
                )
                ok = resp.status_code == 200
                if not ok:
                    logger.warning("sglang update_weights 失败：%s %s", resp.status_code, resp.text[:200])
            except Exception as e:  # noqa: BLE001
                logger.warning("sglang update_weights 异常：%s", e)
                ok = False
        if torch.distributed.is_initialized():
            flag = torch.tensor([int(ok)], device=next(model.parameters()).device)
            torch.distributed.broadcast(flag, src=0)
            torch.distributed.barrier()
            ok = bool(flag.item())
        if not ok:
            raise RuntimeError("SGLang update_policy 失败：引擎权重未同步，训练必须中止（错误三要素：see logs）")


def create_rollout_engine(cfg: RunConfig, policy_model, tokenizer, device: str) -> RolloutEngine:
    if cfg.rl.rollout_engine == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device)
    if cfg.rl.rollout_engine == "sglang":
        return SGLangRolloutEngine(
            cfg.rl.sglang_base_url, cfg.rl.sglang_model_path or "assets/tokenizer", cfg.rl.sglang_shared_path
        )
    raise ValueError(f"未知 rollout 引擎 {cfg.rl.rollout_engine!r}（守门人应已拦截）")

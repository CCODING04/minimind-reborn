"""workflow 共用助手：tokenizer / 模型 / 数据集的统一组装点。"""
from __future__ import annotations

import torch
from transformers import AutoTokenizer

from minimind_reborn import envs
from minimind_reborn.configuration.schemas import RunConfig
from minimind_reborn.data.datasets import DPODataset, PretrainDataset, SFTDataset
from minimind_reborn.data.registry import file_path, resolve_dataset
from minimind_reborn.loggers import get_logger
from minimind_reborn.models.config import config_from_modelcfg
from minimind_reborn.models.model import MiniMindForCausalLM

logger = get_logger("setup")


def load_tokenizer():
    return AutoTokenizer.from_pretrained(envs.tokenizer_path())


def assert_vocab_match(vocab_size: int, tokenizer) -> None:
    """跨源一致性前置断言（llm-inference §1）：词表来自两份产物，必须当场对上。"""
    assert vocab_size == len(tokenizer), (
        f"模型 vocab_size({vocab_size}) != tokenizer 词表({len(tokenizer)})——权重与 tokenizer 混用自不同训练产物"
    )


def build_model(cfg: RunConfig, tokenizer) -> MiniMindForCausalLM:
    assert_vocab_match(cfg.model.vocab_size, tokenizer)
    return MiniMindForCausalLM(config_from_modelcfg(cfg.model))


def build_lm_dataset(cfg: RunConfig, tokenizer, kind: str):
    """按注册表 kind 构建数据集并返回 (train_view, eval_view)。"""
    path = file_path(resolve_dataset(cfg.data.dataset))
    if kind == "pretrain":
        ds = PretrainDataset(path, tokenizer, max_length=cfg.data.max_seq_len, eval_ratio=cfg.data.eval_ratio)
    elif kind == "sft":
        ds = SFTDataset(
            path, tokenizer, max_length=cfg.data.max_seq_len, eval_ratio=cfg.data.eval_ratio,
            add_system_ratio=cfg.data.add_system_ratio, empty_think_ratio=cfg.data.empty_think_ratio,
        )
    else:
        raise ValueError(f"kind {kind!r} 不支持 eval 切分")
    return ds.train_view(), ds.eval_view()


def build_preference_dataset(cfg: RunConfig, tokenizer) -> DPODataset:
    path = file_path(resolve_dataset(cfg.data.dataset))
    return DPODataset(path, tokenizer, max_length=cfg.data.max_seq_len, empty_think_ratio=cfg.data.empty_think_ratio)


def autocast(device: str, dtype: torch.dtype):
    from minimind_reborn.training.common.amp import autocast_context

    return autocast_context(dtype, device)

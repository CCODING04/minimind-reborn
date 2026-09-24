"""按需加载的引擎池（用户决策：同一时刻只驻留一个模型，切换时卸载重载）。

26M fp16 单模型 ~140MB，加载秒级；省显存的代价是对比时逐个加载，
由前端把对比请求拆成顺序生成来消化（见 /api/chat 的 SSE 事件流）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from minimind_reborn.inference.engine import MiniMindLLM
from minimind_reborn.loggers import get_logger
from minimind_reborn.webui.registry import ModelEntry

logger = get_logger("webui.pool")


@dataclass
class LoadedModel:
    model_id: str
    llm: MiniMindLLM


class EnginePool:
    """单槽引擎池：get(entry) 命中当前模型直接复用，否则卸载旧的再加载新的。"""

    def __init__(self, device: str | None = None, dtype: str = "bf16"):
        self._lock = threading.Lock()
        self._current: LoadedModel | None = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype

    @property
    def current_id(self) -> str | None:
        return self._current.model_id if self._current else None

    def get(self, entry: ModelEntry, *, use_best: bool = False) -> MiniMindLLM:
        model_id = entry.model_id + ("@best" if use_best else "")
        with self._lock:
            if self._current is not None and self._current.model_id == model_id:
                return self._current.llm
            self._unload()
            weight = entry.best_path if (use_best and entry.best_path) else entry.weight_path
            logger.info("加载模型 %s（%s）→ %s", model_id, weight, self._device)
            llm = MiniMindLLM.build(
                weight=weight,
                config_path=entry.config_path,
                device=self._device,
                dtype=self._dtype,
            )
            self._current = LoadedModel(model_id=model_id, llm=llm)
            return llm

    def _unload(self) -> None:
        if self._current is None:
            return
        logger.info("卸载模型 %s", self._current.model_id)
        self._current.llm.model.cpu()
        del self._current.llm.model
        self._current = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

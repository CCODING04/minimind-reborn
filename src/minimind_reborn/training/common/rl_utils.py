"""RL 系 workflow 公共件：四件套落盘 / 保存 / 续训（不走 Trainer 基件的循环用这些）。"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from minimind_reborn import env_fingerprint
from minimind_reborn.configuration import snapshot
from minimind_reborn.configuration.schemas import RunConfig
from minimind_reborn.loggers import get_logger, setup_logging
from minimind_reborn.metrics import MetricLogger, build_backends
from minimind_reborn.models.weights import save_inference_weights
from minimind_reborn.training.common.amp import autocast_context, resolve_dtype
from minimind_reborn.training.common.checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
from minimind_reborn.utils import dist
from minimind_reborn.utils.io import atomic_write_json

logger = get_logger("rl")


class RLSession:
    """RL workflow 的公共状态：目录/日志/指标/精度上下文/保存与续训。"""

    def __init__(self, cfg: RunConfig, *, save_weight: str, device: str | None = None, local_rank: int = 0):
        self.cfg = cfg
        self.save_weight = save_weight
        self.local_rank = local_rank
        setup_logging(cfg.run_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dist.is_initialized():
            self.device = f"cuda:{local_rank}"
        self.dtype = resolve_dtype(cfg.train.dtype)
        self.autocast_ctx = autocast_context(self.dtype, self.device)
        self.run_dir = Path(cfg.run_dir)
        self.out_dir = Path(cfg.output_dir)
        self.ckpt_dir = Path(cfg.checkpoint_dir)
        for d in (self.run_dir, self.out_dir, self.ckpt_dir):
            d.mkdir(parents=True, exist_ok=True)
        if dist.is_main_process():
            atomic_write_json(snapshot(cfg), self.run_dir / "config.json")
            env_fingerprint.dump(self.run_dir, seed=cfg.train.seed)
        self.metrics = MetricLogger(
            build_backends(cfg.metrics.backends, self.run_dir, project=cfg.metrics.project, run_name=cfg.recipe_name)
        )

    # ---- 保存（rank0 门控 + best 语义不适用：RL 以 reward 曲线观测，仅存 last） ----
    def save(self, model, optimizer, scheduler, epoch: int, step: int, extra_states: dict | None = None) -> None:
        if not dist.is_main_process():
            return
        weight_path = self.out_dir / self._weight_name()
        save_inference_weights(model, weight_path)
        save_checkpoint(
            checkpoint_path(self.ckpt_dir, self.save_weight, self.cfg.model.hidden_size, self.cfg.model.use_moe),
            model=model,
            optimizer=optimizer,
            scaler=None,  # RL 不用 scaler（bf16 无 scale 语义）
            epoch=epoch,
            step=step,
            config_snapshot=snapshot(self.cfg),
            best_val_loss=None,
            extra_states=extra_states,
        )
        logger.info("已保存 %s（epoch=%d step=%d）", weight_path.name, epoch + 1, step)

    def _weight_name(self) -> str:
        cfg = self.cfg
        return f"{self.save_weight}_{cfg.model.hidden_size}{'_moe' if cfg.model.use_moe else ''}.pth"

    def try_resume(self, model, optimizer, scheduler, extra: dict[str, object] | None = None) -> tuple[int, int]:
        """续训默认行为；extra: {checkpoint key → 带 load_state_dict 的对象}。返回 (start_epoch, start_step)。"""
        cfg = self.cfg
        ckpt = checkpoint_path(self.ckpt_dir, self.save_weight, cfg.model.hidden_size, cfg.model.use_moe)
        if not (cfg.train.resume and ckpt.exists()):
            return 0, 0
        data = load_checkpoint(ckpt, world_size=dist.get_world_size())
        model.load_state_dict(data["model"], strict=True)
        optimizer.load_state_dict(data["optimizer"])
        scheduler.load_state_dict(data["scheduler"])
        for key, obj in (extra or {}).items():
            obj.load_state_dict(data[key])
        logger.info(
            "RL 续训恢复：epoch=%d step=%d（含 %d 个扩展状态）", data["epoch"], data.get("step", 0), len(extra or {})
        )
        return data["epoch"], data.get("step", 0)

    def close(self) -> None:
        self.metrics.close()


def unwrap(model):
    m = model.module if hasattr(model, "module") else model
    return getattr(m, "_orig_mod", m)


def now() -> float:
    return time.time()

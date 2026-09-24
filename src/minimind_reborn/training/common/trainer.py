"""Trainer 基件（training §7 生命周期钩子 + §1-§6 全部不变量的落点）。

train = before → try: epoch 循环 → finally: after。
范式子包只提供 compute_loss(batch, model) -> (loss, log_metrics) 与数据组装。
梯度/AMP/裁剪/续训/评估/保存的顺序与不变量固定在此处，任何范式不得绕过。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from minimind_reborn import env_fingerprint
from minimind_reborn.configuration.schemas import RunConfig
from minimind_reborn.loggers import get_logger
from minimind_reborn.metrics import MetricLogger, build_backends
from minimind_reborn.models.weights import resolve_weight_path, save_inference_weights
from minimind_reborn.training.common.amp import autocast_context, build_scaler, resolve_dtype
from minimind_reborn.training.common.checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
from minimind_reborn.training.common.lr import get_lr
from minimind_reborn.training.common.optim import configure_optimizers
from minimind_reborn.utils import dist
from minimind_reborn.utils.io import atomic_write_json

logger = get_logger("trainer")

LossFn = Callable[[dict, torch.nn.Module], tuple[torch.Tensor, dict[str, float]]]
SaveWeightsFn = Callable[[torch.nn.Module, Path], None]


class SkipBatchSampler(Sampler):
    """跳过前 skip_batches 个 batch（断点在 epoch 中间续训时对齐数据顺序）。

    官方同款语义：外层 sampler 的产出顺序不变，只是前 k 个 batch 不产出——
    因此 worker 也不会加载被跳过的样本。
    """

    def __init__(self, sampler: Sampler | list[int], batch_size: int, skip_batches: int = 0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch: list[int] = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self) -> int:
        total = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total - self.skip_batches)


def _to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class Trainer:
    def __init__(
        self,
        cfg: RunConfig,
        model: torch.nn.Module,
        tokenizer,
        train_ds,
        eval_ds,
        *,
        compute_loss: LossFn,
        save_weight: str,
        device: str | None = None,
        local_rank: int = 0,
        collate_fn=None,
        save_weights_fn: SaveWeightsFn | None = None,
        optimizer_params: list | None = None,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.compute_loss = compute_loss
        self.save_weight = save_weight
        self.local_rank = local_rank
        self.collate_fn = collate_fn
        self.save_weights_fn = save_weights_fn or (lambda m, p: save_inference_weights(m, p))
        self.optimizer_params = optimizer_params

        t = cfg.train
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dist.is_initialized():
            self.device = f"cuda:{local_rank}"
        self.dtype = resolve_dtype(t.dtype)
        self.autocast_ctx = autocast_context(self.dtype, self.device)

        self.run_dir = Path(cfg.run_dir)
        self.out_dir = Path(cfg.output_dir)
        self.ckpt_dir = Path(cfg.checkpoint_dir)
        for d in (self.run_dir, self.out_dir, self.ckpt_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ---- 可追溯四件套：config 快照 + env 指纹（train.log/metrics.jsonl 由 logging/metrics 层写） ----
        if dist.is_main_process():
            from minimind_reborn.configuration import snapshot

            atomic_write_json(snapshot(cfg), self.run_dir / "config.json")
            env_fingerprint.dump(self.run_dir, seed=t.seed)

        self.metrics = MetricLogger(
            build_backends(cfg.metrics.backends, self.run_dir, project=cfg.metrics.project, run_name=cfg.recipe_name)
        )

        self.raw_model = model.to(self.device)
        self.scaler = build_scaler(self.dtype)
        self.optimizer = configure_optimizers(
            self.raw_model,
            t.learning_rate,
            t.weight_decay,
            params=self.optimizer_params if self.optimizer_params is not None else None,
        )
        self.best_val_loss: float | None = None
        self.start_epoch, self.start_step = 0, 0
        self.optimizer_steps = 0

        # ---- 初始化语义三选一（resume > finetune > init_from，互斥由使用场景保证） ----
        self._load_initial_weights()
        # ---- compile 与 DDP 包装（优化器建在裸模型上，之后包装） ----
        if t.compile:
            self.raw_model = torch.compile(self.raw_model)
            logger.info("torch.compile 已启用")
        if dist.is_initialized():
            # 非持久 buffer（RoPE 表）各 rank 由同一 config 确定性算出，广播纯属浪费
            self.model: torch.nn.Module = DistributedDataParallel(
                self.raw_model, device_ids=[local_rank], broadcast_buffers=False
            )
        else:
            self.model = self.raw_model

    # ============ 生命周期 ============
    def run(self) -> None:
        cfg = self.cfg
        total = sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad) / 1e6
        logger.info(
            "训练启动 stage=%s recipe=%s | %.2fM 可训练参数 | dtype=%s device=%s world=%d | 数据=%s",
            cfg.stage,
            cfg.recipe_name,
            total,
            cfg.train.dtype,
            self.device,
            dist.get_world_size(),
            cfg.data.dataset,
        )
        try:
            for epoch in range(self.start_epoch, cfg.train.epochs):
                last_step, exhausted = self._train_epoch(epoch)
                val = self.evaluate()
                if val is not None:
                    improved = self.best_val_loss is None or val < self.best_val_loss
                    if improved:
                        self.best_val_loss = val
                        self._save(epoch, last_step, is_best=True)
                self._save(epoch, last_step)
                if exhausted:
                    break
        finally:
            self.metrics.close()
            logger.info("训练结束：best_val_loss=%s optimizer_steps=%d", self.best_val_loss, self.optimizer_steps)

    def _train_epoch(self, epoch: int) -> None:
        cfg = self.cfg
        t = cfg.train
        loader, iters = self._build_loader(epoch)
        start_step = self.start_step if epoch == self.start_epoch else 0
        if start_step > 0:
            logger.info("Epoch %d：跳过前 %d 个 micro step 续训", epoch + 1, start_step)

        self.model.train()
        window_start = time.time()
        window_tokens = 0
        running_loss = 0.0
        last_grad_norm = 0.0

        for offset, batch in enumerate(loader, start=start_step):
            global_micro = epoch * iters + offset
            total_micro = (
                min(cfg.train.epochs * iters, cfg.train.max_steps)
                if cfg.train.max_steps > 0
                else cfg.train.epochs * iters
            )
            lr = get_lr(
                global_micro, total_micro, t.learning_rate, warmup_steps=t.warmup_steps, min_ratio=t.lr_min_ratio
            )
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            batch = _to_device(batch, self.device)
            is_last = offset == iters - 1
            with self.autocast_ctx:
                loss, log_metrics = self.compute_loss(batch, self.model)
            scaled = loss / t.gradient_accumulation_steps
            self.scaler.scale(scaled).backward()

            # 优化步门：末个 micro step 或累积满 → unscale_ 后裁剪（阈值语义才正确）→ step
            if (offset + 1) % t.gradient_accumulation_steps == 0 or is_last:
                self.scaler.unscale_(self.optimizer)
                params = [p for p in self.model.parameters() if p.requires_grad]
                last_grad_norm = torch.nn.utils.clip_grad_norm_(params, t.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)  # 省内存：引用置 None 不留僵尸梯度
                self.optimizer_steps += 1

            running_loss += loss.item()
            # 范式 batch 结构不同（pretrain 有 input_ids，DPO 是 x/y/mask）——取首个 tensor 的 batch 维
            first_tensor = next(v for v in batch.values() if isinstance(v, torch.Tensor))
            window_tokens += int(first_tensor.shape[0]) * dist.get_world_size() * cfg.data.max_seq_len

            if (offset + 1) % t.log_interval == 0 or is_last:
                elapsed = time.time() - window_start
                tokens_per_s = window_tokens / max(elapsed, 1e-6)
                train_loss = running_loss / t.log_interval
                running_loss = 0.0
                window_start = time.time()
                window_tokens = 0
                eta_min = elapsed / max(offset + 1 - start_step, 1) * (iters - offset - 1) / 60
                stats = {
                    **log_metrics,
                    "train/lr": lr,
                    "train/grad_norm": float(last_grad_norm),
                    "train/tokens_per_s": tokens_per_s,
                }
                self.metrics.log(stats, global_micro)
                logger.info(
                    "Epoch[%d/%d](%d/%d) loss=%.4f lr=%.2e grad_norm=%.2f tok/s=%.0f eta=%.1fmin",
                    epoch + 1,
                    t.epochs,
                    offset + 1,
                    iters,
                    train_loss,
                    lr,
                    float(last_grad_norm),
                    tokens_per_s,
                    eta_min,
                )

            if t.eval_interval_steps > 0 and (offset + 1) % t.eval_interval_steps == 0:
                val = self.evaluate()
                if val is not None and (self.best_val_loss is None or val < self.best_val_loss):
                    self.best_val_loss = val
                    self._save(epoch, offset + 1, is_best=True)

            if (offset + 1) % t.save_interval_steps == 0:
                self._save(epoch, offset + 1)

            if 0 < t.max_steps <= global_micro + 1:
                logger.info("达到 max_steps=%d 截断（冒烟/调试预算）", t.max_steps)
                self._save(epoch, offset + 1)
                return offset + 1, True
        return iters, False

    # ============ 评估（training §7：模板成对出现） ============
    @torch.no_grad()
    def evaluate(self) -> float | None:
        if self.eval_ds is None or len(self.eval_ds) == 0:
            return None
        loader = DataLoader(
            self.eval_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            **_loader_extras(self.cfg.data.num_workers),
        )
        self.model.eval()
        losses: list[float] = []
        for i, batch in enumerate(loader):
            if i >= self.cfg.data.eval_iters:
                break
            batch = _to_device(batch, self.device)
            with self.autocast_ctx:
                loss, _ = self.compute_loss(batch, self.model)
            losses.append(loss.item())
        self.model.train()  # 成对切回，训练态不被评估污染
        # 全 ignore（如 SFT 尾段全是 pad）时 cross_entropy 产生 NaN——过滤而非污染曲线
        finite = [v for v in losses if torch.isfinite(torch.tensor(v))]
        if not finite:
            logger.warning("评估 batch 全部为非有限 loss（验证段全是 ignore token？），本次评估跳过")
            return None
        mean = sum(finite) / len(finite)
        mean = dist.all_reduce_mean(mean, device=self.device)  # 多卡指标聚合，防各卡统计漂移
        self.metrics.log({"val/loss": mean}, self.optimizer_steps)
        logger.info("评估：val/loss=%.4f（固定 %d batch）", mean, len(finite))
        return mean

    # ============ 保存（rank0 独占 + best 另存 + 原子写） ============
    def _save(self, epoch: int, step: int, is_best: bool = False) -> None:
        if not dist.is_main_process():
            return
        from minimind_reborn.configuration import snapshot

        weight_path = self._weight_path()
        self.save_weights_fn(self.model, weight_path)
        if is_best:
            self.save_weights_fn(self.model, self._weight_path(suffix="_best"))
        save_checkpoint(
            checkpoint_path(self.ckpt_dir, self.save_weight, self.cfg.model.hidden_size, self.cfg.model.use_moe),
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            epoch=epoch,
            step=step,
            config_snapshot=snapshot(self.cfg),
            best_val_loss=self.best_val_loss,
        )
        logger.info(
            "已保存：%s%s（epoch=%d step=%d best_val=%s）",
            weight_path.name,
            " + best" if is_best else "",
            epoch + 1,
            step,
            self.best_val_loss,
        )

    def _weight_path(self, suffix: str = "") -> Path:
        cfg = self.cfg
        moe = "_moe" if cfg.model.use_moe else ""
        return self.out_dir / f"{self.save_weight}_{cfg.model.hidden_size}{moe}{suffix}.pth"

    # ============ 数据装载 ============
    def _build_loader(self, epoch: int) -> tuple[DataLoader, int]:
        cfg = self.cfg
        if dist.is_initialized():
            sampler = DistributedSampler(self.train_ds, shuffle=True)
            sampler.set_epoch(epoch)
            inner: Sampler | list[int] = sampler
        else:
            g = torch.Generator()
            g.manual_seed(cfg.train.seed + epoch)  # 采样器显式携带种子（training §1）
            inner = torch.randperm(len(self.train_ds), generator=g).tolist()
        batch_sampler = SkipBatchSampler(
            inner, cfg.train.batch_size, self.start_step if epoch == self.start_epoch else 0
        )
        loader = DataLoader(
            self.train_ds,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            **_loader_extras(cfg.data.num_workers),
        )
        return loader, len(loader)

    # ============ 初始权重 ============
    def _load_initial_weights(self) -> None:
        cfg = self.cfg
        t = cfg.train
        out_root = cfg.output_dir
        if t.finetune_from:
            path = resolve_weight_path(t.finetune_from, out_root, cfg.model.hidden_size, cfg.model.use_moe)
            from minimind_reborn.models.weights import load_finetune_weights

            load_finetune_weights(self.raw_model, path)
            logger.info("微调加载（非严格，缺失 key 已逐条 warning）：%s", path)
        if t.init_from:
            path = resolve_weight_path(t.init_from, out_root, cfg.model.hidden_size, cfg.model.use_moe)
            from minimind_reborn.models.weights import load_inference_weights

            load_inference_weights(self.raw_model, path, strict=True)
            logger.info("严格加载起点权重：%s", path)
        # 铁律 8：续训做成默认行为——checkpoint 在场即恢复
        ckpt = checkpoint_path(self.ckpt_dir, self.save_weight, cfg.model.hidden_size, cfg.model.use_moe)
        if t.resume and ckpt.exists():
            data = load_checkpoint(ckpt, world_size=dist.get_world_size())
            self.raw_model.load_state_dict(data["model"], strict=True)
            self.optimizer.load_state_dict(data["optimizer"])
            self.scaler.load_state_dict(data["scaler"])
            self.start_epoch = data["epoch"]
            self.start_step = data.get("step", 0)
            self.best_val_loss = data.get("best_val_loss")
            logger.info(
                "续训恢复清单：epoch=%d step=%d best_val=%s optimizer=%s scaler=%s（恢复清单让加载错误第一步现形）",
                self.start_epoch,
                self.start_step,
                self.best_val_loss,
                list(data["optimizer"].keys())[:3],
                "ok",
            )


def _loader_extras(num_workers: int) -> dict:
    from minimind_reborn.utils.seed import loader_kwargs

    return loader_kwargs(num_workers)

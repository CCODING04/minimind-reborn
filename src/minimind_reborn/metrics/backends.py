"""指标多后端（logging-metrics §6）。

训练代码只依赖 MetricBackend 协议；本地 jsonl 永远启用（断网/欠费不失忆），
tensorboard/swanlab 可选并列挂载；测试强制 null 后端（Makefile test 目标设置
MINIMID_REBORN_METRICS_DISABLED=1 + metrics=null）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from minimind_reborn import envs
from minimind_reborn.loggers import get_logger, warning_once

logger = get_logger("metrics")


class MetricBackend(Protocol):
    """极小接口：新增后端实现这三个方法即可（鸭子类型，不强制继承）。"""

    def log_scalar(self, key: str, value: float, step: int) -> None: ...

    def log_dict(self, mapping: dict[str, float], step: int) -> None: ...

    def close(self) -> None: ...


class NullBackend:
    """测试与禁用上报场景：吞掉一切。"""

    def log_scalar(self, key: str, value: float, step: int) -> None: ...

    def log_dict(self, mapping: dict[str, float], step: int) -> None: ...

    def close(self) -> None: ...


class JSONLBackend:
    """本地 jsonl 兜底：(step, key, value) 逐行追加，永不可关。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self._fh.write(json.dumps({"step": step, "key": key, "value": value}, ensure_ascii=False) + "\n")

    def log_dict(self, mapping: dict[str, float], step: int) -> None:
        for key, value in mapping.items():
            self.log_scalar(key, value, step)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class TensorBoardBackend:
    """tensorboard 可选后端：惰性 import，未安装时报一次并跳过。"""

    def __init__(self, run_dir: str | Path):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            warning_once(logger, "tb-missing", "tensorboard 未安装，metrics 配置中的 tensorboard 后端被跳过")
            raise
        self.writer = SummaryWriter(log_dir=str(run_dir))

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self.writer.add_scalar(key, value, step)

    def log_dict(self, mapping: dict[str, float], step: int) -> None:
        for key, value in mapping.items():
            self.writer.add_scalar(key, value, step)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


class SwanLabBackend:
    """swanlab 云端后端（可选）：初始化即写入配置快照由调用方负责。"""

    def __init__(self, project: str, run_name: str, run_id: str | None = None):
        import swanlab

        self._swanlab = swanlab
        self.run = swanlab.init(
            project=project, experiment_name=run_name, id=run_id, resume="must" if run_id else None
        )

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self._swanlab.log({key: value}, step=step)

    def log_dict(self, mapping: dict[str, float], step: int) -> None:
        self._swanlab.log(mapping, step=step)

    def close(self) -> None:
        self._swanlab.finish()


_BACKENDS = {
    "null": NullBackend,
    "jsonl": JSONLBackend,
    "tensorboard": TensorBoardBackend,
    "swanlab": SwanLabBackend,
}


def build_backends(
    names: list[str],
    run_dir: str | Path,
    *,
    project: str = "minimind-reborn",
    run_name: str = "run",
    run_id: str | None = None,
) -> list[MetricBackend]:
    """按名字构建后端列表；jsonl 永远追加在首位（本地兜底不依赖配置）。

    环境变量 MINIMIND_REBORN_METRICS_DISABLED=1 时全部替换为 null（测试/CI 强制离线）。
    """
    if envs.metrics_disabled():
        warning_once(logger, "metrics-off", "MINIMIND_REBORN_METRICS_DISABLED=1，全部指标后端替换为 null")
        return [NullBackend()]

    backends: list[MetricBackend] = [JSONLBackend(Path(run_dir) / "metrics.jsonl")]
    for name in names:
        if name == "jsonl":
            continue
        if name not in _BACKENDS:
            raise ValueError(
                f"未知指标后端 '{name}'；可选 {sorted(_BACKENDS)}。"
                f"请检查配方 metrics 字段或环境变量（错误三要素：发生了什么/原因/下一步）"
            )
        kwargs: dict = {}
        if name == "swanlab":
            kwargs = {"project": project, "run_name": run_name, "run_id": run_id}
        try:
            backends.append(_BACKENDS[name](run_dir=run_dir, **kwargs))  # type: ignore[arg-type]
        except ImportError:
            continue  # 惰性 import 失败已在后端内部警告过一次
    return backends


class MetricLogger:
    """训练循环使用的门面：节流（按 log_interval）、rank0 门控、train/val 前缀约定。"""

    def __init__(self, backends: list[MetricBackend]):
        self.backends = backends

    def log(self, mapping: dict[str, float], step: int) -> None:
        for backend in self.backends:
            backend.log_dict(mapping, step)

    def close(self) -> None:
        for backend in self.backends:
            backend.close()

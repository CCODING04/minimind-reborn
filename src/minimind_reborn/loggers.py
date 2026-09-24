"""日志三层职责（logging-metrics §2）。

- 库代码：`get_logger(__name__)` 只 emit 不配置——本模块的 get_logger 就是全项目库代码的入口；
- 应用/训练入口：`setup_logging()` 是唯一做初始化的地方（sink/级别/格式）；
- 平台层能力：once 语义、rank 感知（文件 sink 只挂 rank0）、三方库降噪。

日志与指标是两条通道：事件进日志，数值进 metrics/；训练代码不允许第三种出口。
"""

from __future__ import annotations

import functools
import logging
import sys
import traceback
from pathlib import Path

from minimind_reborn.utils import dist

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_ONCE_SEEN: set[str] = set()
_configured = False


def get_logger(name: str) -> logging.Logger:
    """库代码入口：只获取，不配置。"""
    return logging.getLogger(f"minimind.{name}")


def setup_logging(run_dir: str | Path | None = None, *, level: int = logging.INFO) -> None:
    """训练/推理入口统一调用（幂等）：控制台 + 文件双 sink，文件 sink 只挂 rank0。"""
    global _configured
    if _configured:
        return
    root = logging.getLogger("minimind")
    root.setLevel(level)
    root.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    if run_dir is not None and dist.is_main_process():
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(file_handler)

    quiet_third_party()
    _configured = True


def quiet_third_party() -> None:
    """三方库降噪：已知噪声库压到 WARNING 以上（logging-metrics §2 平台层能力）。"""
    for name in ("transformers", "tokenizers", "httpx", "urllib3", "filelock"):
        logging.getLogger(name).setLevel(logging.WARNING)


def log_once(logger: logging.Logger, level: int, key: str, msg: str) -> None:
    """重复告警只发一次（once 语义，进程级）。"""
    if key in _ONCE_SEEN:
        return
    _ONCE_SEEN.add(key)
    logger.log(level, msg)


def warning_once(logger: logging.Logger, key: str, msg: str) -> None:
    log_once(logger, logging.WARNING, key, msg)


def catch_main(main):
    """入口 main 的异常捕获装饰器：先落日志再抛出（控制台会被刷掉，日志不会）。"""

    @functools.wraps(main)
    def wrapper(*args, **kwargs):
        log = get_logger("main")
        try:
            return main(*args, **kwargs)
        except SystemExit:
            raise
        except BaseException:
            log.error("未捕获异常，完整堆栈如下：\n%s", traceback.format_exc())
            raise

    return wrapper

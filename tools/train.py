"""训练薄入口（project-structure §2.2：只做 解析参数 → 取配置 → 启动）。

用法：
    uv run python tools/train.py configs/pretrain_26m.yaml
    uv run python tools/train.py configs/pretrain_26m.yaml --set train.max_steps=10 --set model.hidden_size=512
"""
from __future__ import annotations

import argparse

import torch

from minimind_reborn.configuration import load_config
from minimind_reborn.loggers import catch_main, get_logger, setup_logging
from minimind_reborn.utils import dist

logger = get_logger("train")

WORKFLOWS = {
    "pretrain": "minimind_reborn.training.pretrain",
    "sft": "minimind_reborn.training.sft",
    "dpo": "minimind_reborn.training.dpo",
    "distill": "minimind_reborn.training.distill",
    "lora": "minimind_reborn.training.lora",
    "grpo": "minimind_reborn.training.grpo",
    "ppo": "minimind_reborn.training.ppo",
    "agent": "minimind_reborn.training.agent",
}


@catch_main
def main() -> None:
    parser = argparse.ArgumentParser(description="minimind_reborn 训练入口（配方 yaml 即实验名）")
    parser.add_argument("recipe", nargs="?", default=None, help="配方文件路径（configs/*.yaml）")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="domain.field=value", help="命令行覆盖（yaml 之后生效，未知 key 报错）")
    args = parser.parse_args()

    local_rank = dist.init_distributed()
    cfg = load_config(args.recipe, args.overrides)
    setup_logging(cfg.run_dir)
    logger.info("配方=%s stage=%s overrides=%s", cfg.recipe_name, cfg.stage, args.overrides)

    import importlib

    workflow = importlib.import_module(WORKFLOWS[cfg.stage])
    workflow.run(cfg, local_rank=local_rank)
    dist.cleanup()


if __name__ == "__main__":
    main()

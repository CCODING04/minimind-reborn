"""模型注册表：从 runs/ 的训练产物自动发现可对话模型。

用户视角：选中 runs 下的某次训练 = 选中一个可推理的模型——
该 run 的 config.json 给出结构（model 域快照）与权重名（按 stage 约定），
扁平 out 布局直接命中权重文件。lora 适配器不含基模权重，v1 不纳入。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minimind_reborn import envs
from minimind_reborn.loggers import get_logger
from minimind_reborn.utils.io import read_json

logger = get_logger("webui.registry")

# stage → 保存权重名前缀（与 Trainer/RLSession 的 save_weight 约定一致）
STAGE_WEIGHT_PREFIX = {
    "pretrain": "pretrain",
    "sft": "full_sft",
    "dpo": "dpo",
    "distill": "full_dist",
    "grpo": "grpo",
    "ppo": "ppo_actor",
    "agent": "agent",
}


@dataclass(frozen=True)
class ModelEntry:
    model_id: str  # 如 full_sft_768
    stage: str
    weight_name: str  # 如 full_sft_768（官方命名）
    weight_path: str
    best_path: str | None  # best 另存副本（若有）
    config_path: str  # run 的 config.json（结构快照来源）
    run_dir: str
    recipe_name: str
    trained_at: str
    hidden_size: int
    use_moe: bool


def _weight_prefix(stage: str, cfg: dict[str, Any]) -> str | None:
    if stage == "lora":
        return None  # 适配器权重需配基模合并，v1 不纳入对话列表
    if stage == "ppo":
        return "ppo_actor"
    if stage == "lora":
        return cfg.get("lora", {}).get("lora_name")
    return STAGE_WEIGHT_PREFIX.get(stage)


def discover(out_root: Path | None = None, run_root: Path | None = None) -> list[ModelEntry]:
    """扫描 runs/（按 config.json 解析）∪ out/（直接匹配权重的兜底），返回去重后的模型目录。"""
    out_root = Path(out_root) if out_root else envs.out_root()
    run_root = Path(run_root) if run_root else envs.run_root()
    found: dict[str, ModelEntry] = {}

    for cfg_path in sorted(run_root.glob("*/*/config.json")):
        try:
            snap = read_json(cfg_path)
        except (OSError, json.JSONDecodeError):
            continue
        stage = snap.get("stage")
        model = snap.get("model", {})
        hidden = model.get("hidden_size")
        if not stage or not hidden:
            continue
        prefix = _weight_prefix(stage, snap)
        if prefix is None:
            continue
        suffix = "_moe" if model.get("use_moe") else ""
        weight_name = f"{prefix}_{hidden}{suffix}"
        out_dir = Path(snap.get("output_dir") or out_root)
        weight_path = out_dir / f"{weight_name}.pth"
        if not weight_path.exists():
            continue
        best = out_dir / f"{weight_name}_best.pth"
        model_id = weight_name
        entry = ModelEntry(
            model_id=model_id,
            stage=stage,
            weight_name=weight_name,
            weight_path=str(weight_path),
            best_path=str(best) if best.exists() else None,
            config_path=str(cfg_path),
            run_dir=str(cfg_path.parent),
            recipe_name=cfg_path.parent.parent.name,
            trained_at=cfg_path.parent.name,
            hidden_size=int(hidden),
            use_moe=bool(model.get("use_moe")),
        )
        found[model_id] = entry  # 同名取字典序最新的 run（glob 已按名排序）

    entries = sorted(found.values(), key=lambda e: e.model_id)
    if not entries:
        logger.warning("runs/ 下未发现可推理的模型产物（需要 config.json + 对应权重的 run）")
    return entries

"""配置装载与守门人（configuration §2 铁律 2/5）。

覆盖顺序固定：dataclass 默认值 → yaml 配方 → 命令行 --set。
任何一层：
- 未知 key 直接报错（防拼写错误静默用默认值训练）；
- 类型不符报错（字符串 "0.001" 覆盖 float 是经典事故）；
- 解析完成后经过唯一守门人 validate_and_derive：互斥检查、跨域依赖检查、派生参数推导。
"""
from __future__ import annotations

import ast
import dataclasses
import datetime as _dt
from pathlib import Path
from typing import Any

from minimind_reborn import envs
from minimind_reborn.configuration.schemas import STAGES, RunConfig

# stage → 必须存在的域（跨域依赖检查的机器可读形式）
_STAGE_DOMAINS: dict[str, tuple[str, ...]] = {
    "pretrain": (),
    "sft": (),
    "dpo": ("dpo",),
    "distill": ("distill",),
    "lora": ("lora",),
    "grpo": ("rl",),
    "ppo": ("rl",),
    "agent": ("rl",),
}


def _domain_instance(cfg: RunConfig, domain: str) -> Any:
    if not hasattr(cfg, domain):
        raise ValueError(f"未知配置域 '{domain}'：合法域为 {sorted(f.name for f in dataclasses.fields(RunConfig))}")
    return getattr(cfg, domain)


def _apply_dict(cfg: RunConfig, mapping: dict[str, Any], source: str) -> None:
    """把嵌套 dict（yaml）合并进 RunConfig；未知 key / 类型不符报错。"""
    for domain, values in mapping.items():
        if domain in ("stage", "recipe_name", "output_dir", "checkpoint_dir", "run_dir"):
            setattr(cfg, domain, values)
            continue
        target = _domain_instance(cfg, domain)
        if not dataclasses.is_dataclass(target):
            raise ValueError(f"配置域 '{domain}' 不是参数对象（yaml 语法层级错误？来自 {source}）")
        field_types = {f.name: f.type for f in dataclasses.fields(target)}
        if not isinstance(values, dict):
            raise ValueError(f"域 '{domain}' 的值应为映射（yaml 缩进错误？来自 {source}）")
        for key, value in values.items():
            if key not in field_types:
                raise ValueError(
                    f"未知配置项 '{domain}.{key}'（来自 {source}）。"
                    f"合法项：{sorted(field_types)}。拼写错误的 key 会静默使用默认值训练，因此直接报错。"
                )
            current = getattr(target, key)
            coerced = _coerce_type(value, current, f"{domain}.{key}")
            setattr(target, key, coerced)


def _coerce_type(value: Any, current: Any, path: str) -> Any:
    """类型断言：yaml/CLI 覆盖值必须与默认值类型一致（bool/int 例外链）。"""
    if current is None:  # 哨兵字段（None 表示自动）不做类型约束
        return value
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        raise ValueError(f"配置项 {path} 需要 bool，得到 {value!r}（{type(value).__name__}）")
    if isinstance(current, int) and not isinstance(current, bool):
        return _check_isinstance(value, int, path, allow_bool=False)
    if isinstance(current, float):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise ValueError(f"配置项 {path} 需要 float，得到 {value!r}——字符串数字不会自动转换，请在 yaml 中写数字字面量")
    if isinstance(current, str):
        return _check_isinstance(value, str, path)
    if isinstance(current, list):
        return _check_isinstance(value, list, path)
    return value


def _check_isinstance(value: Any, tp: type, path: str, *, allow_bool: bool = True) -> Any:
    if isinstance(value, tp) and (allow_bool or not isinstance(value, bool)):
        return value
    raise ValueError(f"配置项 {path} 需要 {tp.__name__}，得到 {value!r}（{type(value).__name__}）")


def _apply_override(cfg: RunConfig, override: str) -> None:
    """--set domain.field=value：点路径定位 + literal_eval 保类型；顶层字段（stage 等）单级即可。"""
    if "=" not in override:
        raise ValueError(f"--set 参数应为 domain.field=value 形式，得到 {override!r}")
    path, raw = override.split("=", 1)
    parts = path.split(".")
    top_level = {"stage", "recipe_name", "output_dir", "checkpoint_dir", "run_dir"}
    if len(parts) == 1:
        if parts[0] not in top_level:
            raise ValueError(f"'{parts[0]}' 是配置域，需要 domain.field 两级路径（如 train.lr=0.001）")
        setattr(cfg, parts[0], raw)
        return
    if len(parts) != 2:
        raise ValueError(f"--set 路径应形如 domain.field（两级），得到 {path!r}")
    domain, field_name = parts
    target = _domain_instance(cfg, domain)
    if domain in top_level:
        setattr(cfg, domain, raw)
        return
    field_types = {f.name: f.type for f in dataclasses.fields(target)}
    if field_name not in field_types:
        raise ValueError(f"未知配置项 '{domain}.{field_name}'。合法项：{sorted(field_types)}")
    current = getattr(target, field_name)
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw  # 字符串字面量：交给类型断言拦下不匹配者
    setattr(target, field_name, _coerce_type(value, current, f"{domain}.{field_name}"))


def load_config(recipe: str | Path | None = None, overrides: list[str] | None = None) -> RunConfig:
    """唯一入口：默认值 → yaml → CLI 三层覆盖后过守门人。

    recipe 为 yaml 路径；experiment 名 = 配方文件名（铁律：实验名可追溯到配方）。
    """
    cfg = RunConfig()
    if recipe is not None:
        recipe_path = Path(recipe).expanduser()
        if not recipe_path.exists():
            raise FileNotFoundError(f"配方文件不存在：{recipe_path}（检查相对路径或用 configs/ 下的现成配方）")
        import yaml

        mapping = yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {}
        _apply_dict(cfg, mapping, source=str(recipe_path))
        cfg.recipe_name = recipe_path.stem
    for override in overrides or []:
        _apply_override(cfg, override)
    return validate_and_derive(cfg)


def validate_and_derive(cfg: RunConfig) -> RunConfig:
    """守门人：互斥/依赖检查 + 派生参数推导（全部配置必须经过此处）。"""
    import math

    if cfg.stage not in STAGES:
        raise ValueError(f"未知训练范式 stage={cfg.stage!r}；可选 {STAGES}")

    # ---- 模型域派生（哨兵 → 推导值） ----
    m = cfg.model
    m.head_dim = m.head_dim or m.hidden_size // m.num_attention_heads
    m.intermediate_size = m.intermediate_size or math.ceil(m.hidden_size * math.pi / 64) * 64
    m.moe_intermediate_size = m.moe_intermediate_size or m.intermediate_size
    if m.num_attention_heads % m.num_key_value_heads != 0:
        raise ValueError(
            f"num_attention_heads({m.num_attention_heads}) 必须被 num_key_value_heads({m.num_key_value_heads}) 整除（GQA 约束）"
        )
    if m.hidden_size % m.num_attention_heads != 0:
        raise ValueError(f"hidden_size({m.hidden_size}) 必须被 num_attention_heads({m.num_attention_heads}) 整除")

    # ---- 训练域校验 ----
    t = cfg.train
    if t.gradient_accumulation_steps < 1:
        raise ValueError(f"gradient_accumulation_steps 必须 ≥1，得到 {t.gradient_accumulation_steps}")
    if t.dtype not in ("fp32", "bf16", "fp16"):
        raise ValueError(f"dtype 只支持 fp32/bf16/fp16，得到 {t.dtype!r}")
    if t.max_steps == 0:
        raise ValueError("max_steps=0 语义歧义（哨兵是 -1 表示不截断）；要跑 0 步请直接不启动训练")
    if t.eval_interval_steps < 0 or t.save_interval_steps < 1:
        raise ValueError(f"eval_interval_steps({t.eval_interval_steps})/save_interval_steps({t.save_interval_steps}) 非法")

    # ---- 数据域校验 ----
    d = cfg.data
    if not (0 <= d.eval_ratio < 1):
        raise ValueError(f"eval_ratio 应在 [0,1)，得到 {d.eval_ratio}")

    # ---- 范式依赖：stage 对应域的参数会被使用，缺依赖即报错（fail-fast 在解析期） ----
    for domain in _STAGE_DOMAINS[cfg.stage]:
        _domain_instance(cfg, domain)  # 存在性由 dataclass 保证；此处触发显式心理检查

    # ---- RL 域校验 ----
    if cfg.stage in ("grpo", "ppo", "agent"):
        if cfg.rl.num_generations < 1:
            raise ValueError(f"rl.num_generations 必须 ≥1，得到 {cfg.rl.num_generations}")
        if cfg.rl.loss_type not in ("cispo", "grpo"):
            raise ValueError(f"rl.loss_type 只支持 cispo/grpo，得到 {cfg.rl.loss_type!r}")
        if cfg.rl.rollout_engine not in ("torch", "sglang"):
            raise ValueError(f"rl.rollout_engine 只支持 torch/sglang，得到 {cfg.rl.rollout_engine!r}")

    # ---- 输出目录派生 ----
    # out/checkpoints 采用扁平布局（{weight}_{hidden}[_moe].pth），跨阶段 init_from 才能按官方命名直接解析；
    # runs/ 按配方名+时间戳嵌套（可追溯四件套的归属）
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.output_dir = cfg.output_dir or str(envs.out_root())
    cfg.checkpoint_dir = cfg.checkpoint_dir or str(envs.checkpoint_root())
    cfg.run_dir = cfg.run_dir or str(envs.run_root() / cfg.recipe_name / stamp)
    return cfg


def snapshot(cfg: RunConfig) -> dict[str, Any]:
    """配置快照：随 checkpoint 与 run_dir/config.json 落盘（铁律 3：只拿到输出目录可唯一重建）。"""
    return dataclasses.asdict(cfg)

# minimind_reborn

MiniMind 的工程化重构：以 `skills/ml-dev-spec` 训练项目形态复现 64M 级中文小模型全流程
（pretrain → SFT → DPO → 蒸馏 / LoRA / GRPO / PPO / Agent RL → 推理服务）。
重构动机与逐条诊断（19 条不满意清单）见 **[重构实录报告](../../Reproduce/report/index.html)**。

## 结果速览

| 项 | 值 |
|---|---|
| 模型 | 64M dense（hidden 768 × 8 层，vocab 6400，与官方 minimind-64M 同规模） |
| 硬件 | 2 × RTX 4090 24GB（torchrun DDP） |
| pretrain 吞吐 | ~190K tokens/s（bf16，seq 340） |
| 测试 | 43 个单元测试（快层 CPU 可跑） |
| 冒烟 | 8 个训练范式 + torchrun 双卡 DDP 全部通过 |

## quick start

```bash
# 0) 环境（uv + CUDA wheel 匹配见 skills/ml-dev-spec/env-dependencies §2）
uv sync --extra dev
uv run python tools/verify_env.py          # 装完 torch 必跑

# 1) 冒烟（tiny 模型，分钟级，验证全链路）
make smoke

# 2) 完整训练（64M，单卡去掉 torchrun 前缀即可）
uv run torchrun --nproc-per-node 2 tools/train.py configs/pretrain_64m.yaml
uv run torchrun --nproc-per-node 2 tools/train.py configs/sft_64m.yaml     # init_from: pretrain
uv run torchrun --nproc-per-node 2 tools/train.py configs/dpo_64m.yaml     # init_from: full_sft

# 3) 对话
uv run python tools/chat.py --weight full_sft
uv run python tools/eval_llm.py --weight dpo   # 自动评测 + 吞吐

# 4) 服务（OpenAI 风格）
uv sync --extra serving
uv run python tools/serve.py --weight dpo --port 8998
```

配方即实验名：`configs/*.yaml` 只写与默认值的差异，命令行 `--set domain.field=value` 可覆盖
（未知 key 直接报错）。续训默认开启——输出目录已有 checkpoint 自动恢复（`--set train.resume=false` 重跑）。

## 数据

默认读取 `MINIMIND_REBORN_DATA_ROOT`（缺省 `~/Code/WorkSpace/minimind-reproduce/dataset`）。
缺失数据集走 modelscope 下载：

```bash
uv sync --extra download
uv run python tools/download_data.py --name rlaif_mini    # 注册表任一名字
```

## 目录结构

```
src/minimind_reborn/
├── configuration/   # 分域 dataclass + yaml/CLI 合并 + 守门人校验
├── data/            # 注册表 + 行偏移索引 jsonl + loss mask + 5 个 Dataset
├── models/          # MiniMind（GQA+QK-Norm+RoPE/MoE，KV cache 预分配）+ LoRA + 权重语义
├── training/        # common 基件（Trainer/AMP/checkpoint 六件套）+ 8 个同构范式子包
├── inference/       # 采样纯函数 → 生成内核（停止三重）→ chat → build 工厂
└── metrics/ loggers.py utils/ envs.py   # 横切层：jsonl 兜底 + 四件套 + 容错封装
```

## 工程目标（Makefile）

```bash
make style       # ruff format + lint --fix（ruff 版本锁定）
make quality     # 只查不改（CI 同源）
make test        # pytest 快层（强制离线，metrics 全部替换 null）
make smoke       # tiny 模型 pretrain→sft→dpo 全链路
make verify-env  # torch/CUDA 验证
```

## 与官方 minimind 的关系

- **state_dict 键级兼容**：官方发布权重可直接 `load_inference_weights` 载入对照复现；
- **tokenizer 完全一致**（官方 6400 词表随包携带），数据格式互通；
- 差异在工程层：配方系统、验证评估、可追溯四件套、测试、KV cache 预分配、采样纯函数化。
  逐条证据见报告 §02 诊断清单。

## 已知短板

诚实清单见报告 §07：web_demo / eval_toolcall 未移植；RL 系验收深度为冒烟级（规则奖励下无学习信号）；
CI 未接 runner（无远端仓库）。

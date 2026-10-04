# minimind_reborn

MiniMind 的工程化重构：以训练项目形态复现 64M 级中文小模型全流程
（pretrain → SFT → DPO → 蒸馏 / LoRA / GRPO / PPO / Agent RL → 推理服务），
并沿「**训练 → 测量 → 归因 → 改数据/管线 → 重训 → 同口径复测**」的改进循环走了完整两轮——
多轮对话跨轮记忆召回 **16.7% → 75.0%**，12 轮长会话末段复读率 **0.454 → 0.086**。
全部实验报告在 [`docs/`](docs/reports.html)（离线单文件 HTML，含完整数字与「预测 vs 实际」分析）。

## 结果速览

| 项 | 值 |
|---|---|
| 模型 | 64M dense（hidden 768 × 8 层，GQA，vocab 6400，与官方 minimind-64M 同构、键级互通） |
| 硬件 | 2 × RTX 4090 24GB（torchrun DDP，无 NVLink，双卡并行效率 88.5%） |
| 训练吞吐 | pretrain ~250K tokens/s；SFT@1536 修复后 185-213K（曾发现并修复编码重复分词 + LR 退火分母 bug，吞吐 ×2） |
| 测试 | 85 个单元测试（CPU 快层）+ ruff/mypy 全绿 |
| 多轮能力（同口径四套件） | 跨轮召回 16.7% → 50.0% → 75.0%（三代模型，超目标线 60%）；末 3 轮复读 0.454 → 0.086 |

## quick start

```bash
# 0) 环境
uv sync --extra dev
uv run python tools/verify_env.py          # 装完 torch 必跑

# 1) 冒烟（tiny 模型，分钟级，验证全链路）
make smoke

# 2) 完整训练（64M，单卡去掉 torchrun 前缀即可）
uv run torchrun --nproc-per-node 2 tools/train.py configs/pretrain_64m.yaml
uv run torchrun --nproc-per-node 2 tools/train.py configs/sft_64m.yaml     # init_from: pretrain
uv run torchrun --nproc-per-node 2 tools/train.py configs/dpo_64m.yaml     # init_from: full_sft

# 3) 对话与评测
uv run python tools/chat.py --weight full_sft
uv run python tools/eval_llm.py --weight dpo
uv run python tools/eval_multiturn.py --suites ABCD --out out/multiturn_eval.md   # 多轮四套件

# 4) 服务（OpenAI 风格）
uv sync --extra serving
uv run python tools/serve.py --weight dpo --port 8998
```

配方即实验名：`configs/*.yaml` 只写与默认值的差异，命令行 `--set domain.field=value` 可覆盖
（未知 key 直接报错）。续训默认开启——输出目录已有 checkpoint 自动恢复（`--set train.resume=false` 重跑）。

## 复现指南：两轮改进的训练链

**档一（数据管线修复 + 窗口 1536）**——修复五项数据处理缺陷（按整轮保留对话 / 半截回答不计损 /
零效果样本过滤 / 回复去重 / 验证集按内容哈希分配）后重训 SFT：

```bash
uv run torchrun --nproc-per-node 2 tools/train.py configs/sft_full_lc.yaml   # seq 1536，~26h
uv run torchrun --nproc-per-node 2 tools/train.py configs/dpo_full.yaml
```

**档二（长上下文续训 + 多轮增强）**——底座窗口 380→1536 续训，再在增强数据上重训：

```bash
# 1) 长文档子集（两段式：字符粗筛 + 逐行分词精筛，自检精度门）
uv run python tools/build_longctx_subset.py

# 2) 外部多轮数据（Belle 83 万清洗漏斗 + 评测探针去污染，抽取 30 万合入）
uv sync --extra download && uv run python tools/fetch_multiturn.py
uv run python tools/merge_multiturn.py

# 3) 底座续训（58min；验收门：长文档 CE 改善 + 短文档防遗忘 ≤5%）
uv run torchrun --nproc-per-node 2 tools/train.py configs/pretrain_longctx.yaml
uv run python tools/eval_cont_train.py

# 4) 增强 SFT（多轮 ×2 / 深多轮 ×3 上采样，~30h）+ DPO
uv run torchrun --nproc-per-node 2 tools/train.py configs/sft_full_mt.yaml
uv run torchrun --nproc-per-node 2 tools/train.py configs/dpo_full.yaml

# 5) 同口径终验收
uv run python tools/eval_multiturn.py --suites ABCD --weights ours_dpo,ours_dpo_era1,official_full_sft
```

## 数据

默认读取 `MINIMIND_REBORN_DATA_ROOT`（缺省 `~/Code/WorkSpace/minimind-reproduce/dataset`）。
缺失数据集走 modelscope 下载（`uv sync --extra download && uv run python tools/download_data.py --name <注册名>`）。
多轮增强数据由 `fetch_multiturn.py`（hf-mirror）+ `merge_multiturn.py`（清洗/去污染/合入）生成，
配套索引（去重/分桶/上采样权重）在首次构建数据集时自动生成。

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/reports.html](docs/reports.html) | **报告主线**：七章节按实验顺序串联全部报告，从这里进 |
| [docs/retrain_plan.html](docs/retrain_plan.html) | 重训优化计划 + 全部实施实录（两轮改进的总账，含预测 vs 实际分析） |
| [docs/model_features_compare.html](docs/model_features_compare.html) | 特征字典（每个指标的意义/算法/测量/陷阱）+ 三代模型对比分析 |
| [docs/dataeng.html](docs/dataeng.html) | 数据工程实测：七维画像、截断损毁定位、五项修复方案 |
| [docs/perf_analysis.html](docs/perf_analysis.html) | 训练吞吐瓶颈与 LR 退火 bug 的完整排查与修复 |
| [docs/multiturn.html](docs/multiturn.html) | 多轮能力四套件检测（评测方法定义处） |
| [docs/hardening.html](docs/hardening.html) / [docs/index.html](docs/index.html) | 加固实录与重构实录（19 条诊断、官方权重对照） |
| [docs/pretrain_readiness.md](docs/pretrain_readiness.md) | 全量训练前置实测（断点续训/双卡/温度） |
| [docs/results/](docs/results/) | 全部实验产物（评测/画像/数据工程日志）+ 索引说明 |

## 目录结构

```
src/minimind_reborn/
├── configuration/   # 分域 dataclass + yaml/CLI 合并 + 守门人校验
├── data/            # 注册表 + 行偏移索引 jsonl + manifest 配套索引 + loss mask + 轮次边界截断
├── models/          # MiniMind（GQA+QK-Norm+RoPE/MoE，YaRN 单源化）+ LoRA + 权重语义
├── training/        # common 基件（Trainer/AMP/checkpoint 六件套）+ 8 个同构范式子包
├── inference/       # 采样纯函数 → 生成内核 → chat（历史预算滑窗）→ build 工厂 + 长度硬校验
└── metrics/ loggers.py utils/ envs.py   # 横切层：jsonl 兜底 + 四件套 + 容错封装
```

## 工程目标（Makefile）

```bash
make style       # ruff format + lint --fix（ruff 版本锁定）
make quality     # 只查不改（CI 同源）
make test        # pytest 快层（强制离线）
make smoke       # tiny 模型 pretrain→sft→dpo 全链路
make verify-env  # torch/CUDA 验证
```

## 已知短板与下一步（诚实清单）

- **长上下文生成质量**：合成长历史场景下仍是敷衍短句（11 token）+ 一处单点复读——已归因为
  「训练分布缺失」而非能力不足（底座长文档 CE 已改善 20.7% 而该指标不动），下一步靠合成对话数据
  与生成长度偏好对（方案见 [docs/model_features_compare.html](docs/model_features_compare.html) §6）；
- **跨轮倾听未覆盖全部轮位**：第 2 轮及以后注入的信息（爱好/宠物类）仍会丢，召回 75% → 85% 的
  数据方案已列；
- web_demo / eval_toolcall 未移植；RL 系验收深度为冒烟级；
- 工程小遗留：数据卡的 worker 侧计数未汇总、评测脚本侧超长输入守卫。

## 与官方 minimind 的关系

- **state_dict 键级兼容**：官方发布权重可直接 `load_inference_weights` 载入对照复现；
- **tokenizer 完全一致**（官方 6400 词表随包携带），数据格式互通；
- 差异在工程层：配方系统、验证评估、可追溯四件套、测试、数据管线（去重/分桶/轮次截断/上采样）。
  逐条证据见 [docs/index.html](docs/index.html) 诊断清单。

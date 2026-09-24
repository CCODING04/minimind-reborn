# 检查器清单由本文件拥有（版本锁定）；CI 与本地同源（python-style §1.5-1.6）
RUFF_VERSION := 0.14.0
UV := uv
PYTHON := $(UV) run python

.PHONY: style quality test smoke verify-env download-rl format

style:  ## 自动修复：format + lint --fix
	$(UV) run ruff@$(RUFF_VERSION) format .
	$(UV) run ruff@$(RUFF_VERSION) check --fix .

quality:  ## 只查不改（CI 同源）
	$(UV) run ruff@$(RUFF_VERSION) format --check .
	$(UV) run ruff@$(RUFF_VERSION) check .

test:  ## 测试：强制离线（metrics 全部替换 null，禁云端上报）
	MINIMID_REBORN_METRICS_DISABLED=1 MINIMIND_REBORN_METRICS_DISABLED=1 \
	$(PYTHON) -m pytest tests/ -m "not slow" $(PYTEST_ARGS)

smoke:  ## tiny 模型全链路冒烟（pretrain→sft→dpo）
	$(PYTHON) tools/train.py configs/smoke/smoke_pretrain.yaml
	$(PYTHON) tools/train.py configs/smoke/smoke_sft.yaml
	$(PYTHON) tools/train.py configs/smoke/smoke_dpo.yaml

verify-env:  ## torch/CUDA 环境验证（装完 torch 必跑）
	$(PYTHON) tools/verify_env.py

download-rl:  ## 下载 RL 冒烟所需数据集（modelscope）
	$(PYTHON) tools/download_data.py --name rlaif_mini
	$(PYTHON) tools/download_data.py --name agent_rl_mini

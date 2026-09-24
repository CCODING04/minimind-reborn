#!/bin/bash
# 全量训练链：数据下载校验 → pretrain → SFT → DPO（2×GPU DDP，resume 默认开）
# 用法：nohup bash tools/run_full.sh > /tmp/full_chain.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
UV=${UV:-$HOME/.local/bin/uv}
NP=${NP:-2}
MARKER=/tmp/full_chain.log

echo "[chain] start $(date)" >> "$MARKER"
$UV run python tools/download_data.py --name pretrain_t2t >> "$MARKER" 2>&1 || { echo "[chain] 数据下载失败" >> "$MARKER"; exit 1; }
$UV run python tools/download_data.py --name sft_t2t    >> "$MARKER" 2>&1 || { echo "[chain] 数据下载失败" >> "$MARKER"; exit 1; }

$UV run torchrun --nproc-per-node "$NP" tools/train.py configs/pretrain_full.yaml > /tmp/full_pretrain.log 2>&1
echo "[chain] pretrain exit=$? $(date)" >> "$MARKER"
$UV run torchrun --nproc-per-node "$NP" tools/train.py configs/sft_full.yaml     > /tmp/full_sft.log 2>&1
echo "[chain] sft exit=$? $(date)" >> "$MARKER"
$UV run torchrun --nproc-per-node "$NP" tools/train.py configs/dpo_full.yaml     > /tmp/full_dpo.log 2>&1
echo "[chain] dpo exit=$? $(date)" >> "$MARKER"
echo "[chain] all done $(date)" >> "$MARKER"

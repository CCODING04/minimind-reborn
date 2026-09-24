# 全量训练前置实测报告（pretrain readiness）

日期：2026-09-24 · 环境：i9-14900KF（24C32T）/ 31G 内存 / RTX 4090 + RTX 4090 D（PHB/PCIe 4.0，无 NVLink）/ 787G 磁盘可用
负载基准：64M dense（768×8）、seq 340、bs 32、accum 8、bf16、seed 42、pretrain_t2t_mini

## 1. 断点续训有效性：✅ 通过（且修复 1 个真实 bug）

| 项 | 基线 A（连续 200 步） | 中断 B1（kill -9 @110） | 续训 B2（ckpt@100 → 200） |
|---|---|---|---|
| 恢复清单 | — | — | epoch=0 step=100，optimizer/scaler ✓ |
| step 110 / 150 / 200 loss | 7.4274 / 7.3323 / 7.2656 | —（死于此区间） | 7.3935 / 7.2880 / 7.2276 |
| LR 全程 | 2.43e-4 → 5e-5 | — | **与 A 逐步完全一致** |

- 最大 loss 相对偏差 **0.67%**（判定线 2%），来源为 embedding backward 原子操作的非确定性，无轨迹发散。
- `kill -9` 后无 `.tmp` 残留（原子写验证）、checkpoint 正常加载。
- **实测发现并修复 1 个续训记账 bug**：SkipBatchSampler 会把 `len(loader)` 扣减跳过批数，
  导致续训后全局步数分母 / LR 总步数 / 尾批 `is_last` 判定全部漂移（尾批部分累积会被丢弃）。
  修复：Trainer 改用"完整 epoch 步数"记账 + `is_last` 按 `start_step + iters - 1` 判定。
- 结论：**长跑中断保险成立**；建议全量训练保持 `save_interval_steps: 1000`（约每 8.5 分钟一个 resume 点）。

## 2. DataLoader worker 数量：✅ 2 个即饱和

| num_workers | 0 | 2 | 4 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| 训练稳态 tok/s | 130.0K | **140.5K** | 140.4K | 140.4K | 140.3K | 140.3K | 140.3K |
| loader-only tok/s | 1735K | — | 6425K | 8350K | — | 16442K | — |

- 数据侧上限是训练消耗的 **12× 以上**——瓶颈纯在 GPU 计算，分词（HF fast tokenizer）非常轻。
- **建议：num_workers=4**（2 已饱和，4 留余量；8+ 无收益且占内存/进程）。
- 温度：满载训练 GPU 53-54°C / SM 2520-2775MHz，**无降频**；24h 连续跑无热风险（该负载下单卡 ~85W）。

## 3. 单卡 vs 双卡：✅ 双卡值得（no_sync 修复后 scaling 88.5%）

| 配置 | 稳态 tok/s | scaling 效率 |
|---|---|---|
| 单卡 | 140,760 | — |
| 双卡 DDP（修复前：每 micro step 全量 allreduce） | 190,244 | 67.6% |
| 双卡 DDP（**no_sync 修复后**：仅末个累积步同步） | **249,052** | **88.5%** |

- **实测发现并修复 1 个训练性能 bug**：原 Trainer 在梯度累积时每个 micro step 都做全量 allreduce
  （64M×2B=136MB/步走 PCIe PHB），违反 training §3"仅末个 micro step 同步"。补 `no_sync` 后吞吐 +31%。
- 剩余 11.5% 损耗来源：PHB/PCIe（无 NVLink）+ 混合卡（4090 D 慢 5-7%，DDP 同步点等慢卡）。

### 任务级建议

| 任务 | 建议 | 原因 |
|---|---|---|
| pretrain | **双卡**（accum 8 受 no_sync 修复收益最大） | 纯计算密集，88.5% scaling 直接省 ~6h |
| SFT（full data） | **双卡**（accum 1，无累积收益但数据并行照常） | 墙钟减半；显存充裕 |
| DPO | **单卡足够，双卡省墙钟** | policy+ref ≈1.5-2× 计算、显存 2×权重（64M 单卡无压力）；accum=1 时 no_sync 无收益 |
| GRPO/agent | 单卡可跑；双卡按 rank 分 prompt 近线性 | rollout 生成为主，63M 权重两份驻留无压力 |

## 4. 其他关键事项（全量训练前检查单）

1. **数据下载前置**：pretrain_t2t.jsonl 10GB + sft_t2t.jsonl 14GB，modelscope 缓存峰值双倍磁盘（~48GB，787G 可用无压力），预估 40-60 分钟。
2. **注册表**：`data/datasets.json` 需补 `pretrain_t2t` / `sft_t2t` 两条 full 数据条目。
3. **seq_len**：官方 README 推荐 full 版 pretrain_t2t 用 ≈380（mini 用 ≈768）；我们 mini 配方 340 接近，full 配方建议 380。
4. **checkpoint 写放大**：save_interval 1000 步 ≈ 每 8.5 分钟覆盖写 731MB fp32 resume（NVMe 无压力，28h 约 200 次覆盖）。
5. **长跑保险**：resume 默认开（本次 kill -9 实测通过）；跨卡数重启会自动换算 step（1↔2 卡）。
6. **时长估算**（no_sync 修复后 249K tok/s）：pretrain full 2ep ≈ 8.1h + SFT full 2ep ≈ 12.4h + DPO ≈ 5min ≈ **20.5h 连续**；1 epoch 版 ≈ 10.5h。
7. **散热/供电**：该负载双卡 ~170W 合计（64M 太小），24h 无热风险；若未来换大模型注意机箱散热与 1100W+ 峰值供电。
8. **服务共存**：GPU0 上的推理服务（~650MB）可与训练共存；长训练期间建议避免同时跑大模型推理。
9. **评估口径**：full 数据的 val 切片（0.5%）≈ 15 万样本，eval_iters=50 batch 固定不变保证跨 run 可比。

## 5. 运行时温度实测（2026-09-24，双卡满载 ~5.5 分钟至热稳态）

| 项 | GPU0 (RTX 4090) | GPU1 (RTX 4090 D) |
|---|---|---|
| 热稳态温度 | **76-78°C**（68°C 起，3 分钟进入平台） | **61-63°C** |
| 功耗 | 386-409W | 340-349W |
| SM 频率 | 全程 2745MHz（**无降频**） | 全程 2685MHz（无降频） |
| 持续吞吐 | 253.7K tok/s 稳定 | — |

结论：**温度可控**。4090 系降频阈值约 83-88°C，稳态 78°C 留有 ~10°C 余量且曲线无爬升趋势；
当前 64M 模型 + seq 340 的负载下 24h 连续训练无热风险。注意：若未来加大 seq_len/batch 使功耗逼近
450W 上限，GPU0 稳态温度会进一步上探，建议长跑前复测一次（采样命令已在本节，`nvidia-smi` 每 15s）。

## 6. 实训补充（2026-09-25 全量训练启动时发现）

**"续训默认开"在换数据集场景的副作用**：checkpoint 仅按权重名索引（`{weight}_{hidden}[_moe]_resume.pth`），
不含数据集/配方身份。从 mini 数据切到 full 数据时，残留的 mini-era resume 会被自动加载——
训练"看起来正常恢复"，实际只跑了剩余进度（等效于 mini 2 轮 + full 半轮）。

**处置**：全量启动前将三个 mini-era resume 归档至 `checkpoints/mini_era/`，从零重启（kill -9 中断链 + 归档 + 重启，损失 ~1.5 分钟）。

**规范沉淀**：切换数据集/配方启动训练前的检查单——
1. `ls checkpoints/` 确认无同名残留 resume，或显式归档；
2. 后续改进方向：checkpoint 内嵌 config 快照比对（数据集身份不一致时 warning + 要求显式确认），已列入待办。

# 实验产物索引（docs/results/）

全部产物为同口径实测（seed 42），对应结论见 `../retrain_plan.html` 与 `../model_features_compare.html`。

## 四套件评测（tools/eval_multiturn.py）

| 文件 | 内容 | 对应结论 |
|---|---|---|
| multiturn_eval.md | era1 基线四套件（我方 dpo/full_sft + official） | 旧管线召回 16.7%、末 3 轮重复 0.454、超窗塌缩、>32K 裸崩溃 |
| multiturn_eval_b_baseline.md | 阶段 0 不开外推的 B 套件复测（口径对照） | YaRN 修正前的基线 |
| multiturn_eval_yarn.md | 阶段 0 开 YaRN 外推的 B 套件 | 纯外推负收益（1536/2048 档恶化成复读循环 0.84/0.98） |
| multiturn_eval_v2.md | 档一验收（新 dpo vs era1 vs official） | 召回 16.7%→50.0%、末 3 轮 0.454→0.283、B 不变 |
| multiturn_eval_v3.md | 档二终验收（同三方对照） | 召回 75.0%、末 3 轮 0.086、B 5/6（512 档单点复读） |
| multiturn_samples/ | 上述评测的逐用例全文（含 Suite A 命中归因素材） | "倾听用户"归因：算术用例 17/35/900 首次全中 |

## 数据画像（tools/len_profile.py / data_profile.py）

| 文件 | 内容 | 对应结论 |
|---|---|---|
| len_profile_sft_768_v3.json | C1-C5 修复后 @768 窗复测 | 残段 0.82→0.13、留存区间完整率 93.5%、零损失样本 13.4%（训练期过滤） |
| len_profile_sft_1536_v3.json | 同上 @1536 窗 | 档一训练窗口的定标依据 |
| len_profile_sft_mt_1536.json | 档二多轮增强数据 @1536 | 完整率 89.2% 与档一持平（Belle 无适配负担） |
| data_profile_sft_mt.json | 合并数据字符级画像 | 多轮率 23.9%、回复重复 7.85%（键级去重后 0）、轮次分布 |
| val_baseline_v2.json | 旧权重在新验证集 vs 旧尾切验证集 | 1.2323 vs 2.0419——旧验证分数虚高之谜的实证 |

## 数据工程日志

| 文件 | 内容 | 对应结论 |
|---|---|---|
| merge_multiturn.log | Belle 83.1 万 → 30 万过滤漏斗 | 解析通过 99.6%、去污染剔除 3,424 |
| build_longctx.log | 长文档子集两段式构建 | 首版字符筛精度 56% 被门禁拦截 → 两段式 100%；≥768 tok 实测仅 33 万条 |
| manifest_mt_build.log | 合并数据配套索引统计 | 去重 373,746、分桶差 0.05pp、多轮有效占比 41.7% |

## 续训验收（tools/eval_cont_train.py，结果原在会话记录、此处补记）

```json
{
  "short_docs": {"old_base": 4.5115, "new_base": 4.7046, "delta_pct": 4.28},
  "long_docs":  {"old_base": 4.1208, "new_base": 3.2681, "delta_pct": -20.69},
  "gate": {"forgetting_ok": true, "long_improved": true}
}
```

长文档交叉熵 −20.7%（续训直接目标兑现）；短文档劣化 +4.28% ≤5% 防遗忘门（25% 回放起效）。

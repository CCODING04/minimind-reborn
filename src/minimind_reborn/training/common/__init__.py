"""训练公共基件：生命周期/AMP/优化器/LR/检查点。

原版 train_pretrain/train_full_sft/train_dpo/train_distillation/train_lora 五个脚本
复制的 ~300 行样板全部收拢在这里；范式子包只写差异（compute_loss + 数据组装）。
"""

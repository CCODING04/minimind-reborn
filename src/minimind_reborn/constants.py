"""全局常量：特殊 token、模型默认值。

只放"代码自身无法推导"的固定语义；可调参数一律走 configuration。
"""

from __future__ import annotations

# ===== 特殊 token（与官方 minimind tokenizer 完全一致，保证权重与数据互通） =====
PAD_TEXT = "<|endoftext|>"
BOS_TEXT = "<|im_start|>"
EOS_TEXT = "<|im_end|>"

# ===== SFT/DPO loss mask 的区间标记模式 =====
# chat template 渲染后 assistant 回答总是夹在 "<|im_start|>assistant\n" 与 "<|im_end|>\n" 之间，
# loss mask 依赖对这两个 token 序列的模式匹配（data/loss_mask.py），这是全系统最脆的耦合点，
# 由 tests/test_data/test_loss_mask.py 逐 token 断言锁定。
ASSISTANT_PREFIX_TEXT = f"{BOS_TEXT}assistant\n"
RESPONSE_SUFFIX_TEXT = f"{EOS_TEXT}\n"

"""模型层（project-structure §3 模型文件解剖顺序）。

config → 组合子模块（layers/attention/feedforward，各自独立成类，是测试与
attention 后端注入的挂载点）→ block → 主模型 → 任务模型（ForCausalLM）。

与官方 minimind 的关键差异：
- KV cache 由"每步 torch.cat 重新分配的 tuple 列表"改为一次性预分配 + 原地切片
  （kv_cache.py，llm-inference §2 不变量）；
- generate() 不再长在模型类里——采样进 inference/（内核与文本分层）；
- 敏感算子 fp32 纪律（RMSNorm/softmax/rotary）与 state_dict 键名完全保留官方
  语义，官方发布权重可直接载入对照复现。
"""

from minimind_reborn.models.config import MiniMindConfig, config_from_modelcfg
from minimind_reborn.models.kv_cache import KVCache
from minimind_reborn.models.model import MiniMindBlock, MiniMindForCausalLM, MiniMindModel

__all__ = [
    "KVCache",
    "MiniMindBlock",
    "MiniMindConfig",
    "MiniMindForCausalLM",
    "MiniMindModel",
    "config_from_modelcfg",
]

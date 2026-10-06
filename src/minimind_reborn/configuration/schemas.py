"""配置分域 dataclass（configuration §1 模式 C：生产训练项目）。

按变化原因分域（铁律 4）：一个类只因为一个原因而修改——
model（结构）/ data（数据与切分）/ train（优化与流程）/ metrics（上报）/
generate（采样）/ 各训练范式的专属参数（dpo/distill/lora/rl）。

铁律 1：全部超参集中在此、带默认值、带一句话注释；
铁律 6：必须由运行时注入/派生的参数用 None/-1 哨兵表达，守门人统一推导。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 训练范式清单：配方 stage 字段的合法值（未知值在守门人处报错）
STAGES = ("pretrain", "sft", "dpo", "distill", "lora", "grpo", "ppo", "agent")


@dataclass
class ModelConfig:
    """模型结构参数。resume 时以 checkpoint 内快照为准，正则超参允许覆盖。"""

    hidden_size: int = 768
    num_hidden_layers: int = 8
    use_moe: bool = False
    vocab_size: int = 6400  # 官方 tokenizer 词表；运行时与 tokenizer 双源断言
    num_attention_heads: int = 8
    num_key_value_heads: int = 4  # 等于 num_attention_heads 即 MHA；n<n_q 即 GQA（yaml/--set 只收 int，None 仅 MiniMindConfig Python API 支持）
    head_dim: int | None = None  # 派生：hidden_size // num_attention_heads（哨兵）
    intermediate_size: int | None = None  # 派生：ceil(hidden_size·π/64)·64（官方公式）
    dropout: float = 0.0
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    tie_word_embeddings: bool = True
    inference_rope_scaling: bool = False  # 推理期 YaRN 外推开关，训练恒为 False
    inference_rope_factor: int = 6  # 外推倍数 = 目标窗/训练窗（380×6=2280，覆盖评测 2048 档）
    # ---- MoE 专属（use_moe=False 时忽略） ----
    num_experts: int = 4
    num_experts_per_tok: int = 1
    moe_intermediate_size: int | None = None  # 派生：默认取 intermediate_size
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 5e-4


@dataclass
class DataConfig:
    """数据与切分参数。数据本身不进 git，根目录走环境变量（envs.data_root）。"""

    dataset: str = "pretrain_t2t_mini"  # 注册表名（data/registry.py），未注册名报错
    max_seq_len: int = 340  # 中文 1 token ≈ 1.5~1.7 字符
    num_workers: int = 8
    eval_ratio: float = 0.01  # 数据尾部固定切片作验证集（保证可复现）；0 = 关闭评估
    eval_iters: int = 50  # 每次评估固定 batch 数——评估数据固定，指标才可比
    add_system_ratio: float = 0.2  # SFT 增强：概率插入 system 消息
    empty_think_ratio: float = 0.2  # SFT 增强：概率保留空思考标签
    thinking_ratio: float = 0.5  # RL 数据：概率开启 thinking 前缀


@dataclass
class TrainConfig:
    """优化与训练流程参数（跨范式共享）。"""

    epochs: int = 2
    max_steps: int = -1  # 哨兵：>0 时优先于 epochs 截断总优化步数（冒烟/调试用）
    batch_size: int = 32
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.01  # torch AdamW 默认值；原版脚本隐式使用，这里显式声明
    grad_clip: float = 1.0
    dtype: str = "bf16"  # fp32 | bf16 | fp16（精度即配置项，不散落硬编码）
    seed: int = 42
    deterministic: bool = False  # True 开启 cudnn 确定性（有减速代价，复现实验才开）
    log_interval: int = 10
    eval_interval_steps: int = 0  # >0 时按步评估；0 = 仅 epoch 末评估
    save_interval_steps: int = 1000
    warmup_steps: int = 0  # 官方余弦无 warmup；>0 时线性爬升
    lr_min_ratio: float = 0.1  # 余弦地板：lr_min = lr * 0.1（官方 get_lr 语义）
    compile: bool = False
    resume: bool = True  # 铁律 8：输出目录已有 checkpoint 自动恢复
    finetune_from: str | None = None  # 非严格加载的权重名/路径（缺失 key 逐条 warning）
    init_from: str | None = None  # 严格加载的起点权重名/路径（如 sft ← pretrain）


@dataclass
class MetricsConfig:
    """指标上报（jsonl 兜底永远启用，此处只列附加后端）。"""

    backends: list[str] = field(default_factory=lambda: ["jsonl"])
    project: str = "minimind-reborn"


@dataclass
class GenerateConfig:
    """采样参数（与训练参数分域）。默认取官方推荐值（constants.RECOMMENDED_*）。"""

    temperature: float = 0.85  # ≤0 显式走贪心分支（argmax）
    top_p: float = 0.95
    top_k: int = 50  # 0 = 关闭
    repetition_penalty: float = 1.0  # 1.0 = 关闭
    max_new_tokens: int = 1024


@dataclass
class DPOConfig:
    """DPO 专属参数（train_dpo.py 的 argparse 域）。"""

    beta: float = 0.15  # 偏离参考模型的强度；官方建议 lr ≤ 5e-8 防遗忘


@dataclass
class DistillConfig:
    """蒸馏专属参数（train_distillation.py 的 argparse 域）。"""

    teacher_hidden_size: int = 768
    teacher_num_layers: int = 8
    teacher_use_moe: bool = False
    teacher_weight: str = "pretrain"  # out_root 下的教师权重名
    alpha: float = 0.5  # 总损失 = alpha·CE + (1-alpha)·KL
    temperature: float = 1.5  # 推荐范围 1.0~2.0


@dataclass
class LoRAConfig:
    """LoRA 专属参数（train_lora.py + model_lora.py 的 argparse 域）。"""

    rank: int = 16
    lora_name: str = "lora_reborn"  # 保存的适配器权重名


@dataclass
class RLConfig:
    """RL 系（grpo/ppo/agent）共享参数 + PPO/agent 专属字段。"""

    num_generations: int = 6  # 每个 prompt 的采样组大小（GRPO/agent；PPO 恒为 1）
    beta: float = 0.1  # KL 惩罚系数
    loss_type: str = "cispo"  # cispo | grpo
    epsilon: float = 0.2  # clip 下界偏移
    epsilon_high: float = 5.0  # cispo 上界
    max_gen_len: int = 256  # 冒烟友好默认；原版 1024
    max_total_len: int = 2500  # agent 训练侧总长度上界
    max_turns: int = 3  # agent 多轮工具调用轮数
    thinking_ratio: float = 0.9
    reward_model_path: str = ""  # 空 = 只用规则奖励（跳过外部 reward model）
    rollout_engine: str = "torch"  # torch | sglang
    sglang_base_url: str = "http://localhost:8998"
    sglang_model_path: str = ""
    sglang_shared_path: str = "sglang_ckpt"
    # ---- PPO 专属 ----
    clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    kl_coef: float = 0.02
    gamma: float = 1.0
    lam: float = 0.95
    cliprange_value: float = 0.2
    ppo_update_iters: int = 2
    early_stop_kl: float = 0.25
    mini_batch_size: int = 2
    critic_learning_rate: float = 5e-7


@dataclass
class RunConfig:
    """一次实验的完整定义：配方文件（yaml）按域映射到这些字段。"""

    stage: str = "pretrain"
    recipe_name: str = "default"  # 实验名 = 配方文件名（指标曲线 ↔ 输出目录对齐）
    output_dir: str = ""  # 派生：out_root()；"" 为装载前哨兵（守门人必填）
    checkpoint_dir: str = ""  # 派生：checkpoint_root()
    run_dir: str = ""  # 派生：run_root()/recipe_name/时间戳
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    rl: RLConfig = field(default_factory=RLConfig)

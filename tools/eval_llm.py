"""自动评测：固定问题集 + 吞吐统计（官方 eval_llm.py 的重构）。"""
from __future__ import annotations

import argparse

from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.inference.engine import MiniMindLLM

PROMPTS = [
    "你有什么特长？",
    "为什么天空是蓝色的",
    "请用Python写一个计算斐波那契数列的函数",
    "解释一下\"光合作用\"的基本过程",
    "如果明天下雨，我应该如何出门",
    "比较一下猫和狗作为宠物的优缺点",
    "解释什么是机器学习",
    "推荐一些中国的美食",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="minimind_reborn 自动评测")
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--pretrain-mode", action="store_true", help="base 模型走 bos+text 直拼")
    args = parser.parse_args()

    llm = MiniMindLLM.build(args.weight, hidden_size=args.hidden_size,
                            num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    gen = GenerateConfig(temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens)
    total_tokens = total_time = 0.0
    for prompt in PROMPTS:
        result = llm.generate_text([{"role": "user", "content": prompt}], gen, pretrain_mode=args.pretrain_mode)
        total_tokens += result["generated_tokens"]
        total_time += result["latency_ms"] / 1000
        print(f"💬: {prompt}\n🧠: {result['content']}"
              + (f"\n（思考：{result['reasoning_content'][:80]}…）" if result["reasoning_content"] else "")
              + f"\n[{result['generated_tokens']} tok | {result['latency_ms']:.0f}ms | {result['finish_reason']}]\n")
    print(f"平均吞吐: {total_tokens / max(total_time, 1e-6):.1f} tokens/s")


if __name__ == "__main__":
    main()

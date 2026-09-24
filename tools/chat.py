"""交互对话入口：封装 MiniMindLLM（build 工厂），支持 thinking 开关与速度统计。"""
from __future__ import annotations

import argparse
import random
import time

from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.inference.engine import MiniMindLLM


def main() -> None:
    parser = argparse.ArgumentParser(description="minimind_reborn 交互对话")
    parser.add_argument("--weight", default="full_sft", help="权重名（pretrain/full_sft/dpo/...）")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--open-thinking", action="store_true", help="开启 <think> 思考前缀")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    llm = MiniMindLLM.build(
        args.weight, hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe, device=args.device,
    )
    gen = GenerateConfig(temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens)
    pretrain_mode = "pretrain" in args.weight
    session = llm.chat_session(gen, pretrain_mode=pretrain_mode)
    print("输入问题开始对话（Ctrl-C 退出）")
    while True:
        try:
            query = input("💬: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        random.seed(random.randint(0, 2**31))  # 采样语义：默认非零温度下两次输出不同是正常行为
        start = time.time()
        text = session.respond(query, open_thinking=args.open_thinking)
        tokens = len(session.tokenizer(text).input_ids)
        print(f"🧠: {text}\n[速度] {tokens / max(time.time() - start, 1e-6):.1f} tokens/s\n")


if __name__ == "__main__":
    main()

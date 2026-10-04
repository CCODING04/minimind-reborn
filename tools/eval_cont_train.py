"""续训验收门（重训计划档二 4a）：防遗忘 + 长文本改善，双指标对比新旧底座。

指标 A（防遗忘）：随机短文档（<340 token）上新旧底座的交叉熵，新底座劣化 ≤5%。
指标 B（长文本改善）：随机长文档（≥1024 token，1536 窗口）上新旧底座的交叉熵，
新底座应显著更低（续训的直接目标——底座在长位置上的语言建模能力）。

用法：uv run python tools/eval_cont_train.py [--n 200]
"""

from __future__ import annotations

import argparse
import json
import random

import torch


@torch.no_grad()
def ce_loss(model, tokenizer, texts: list[str], max_length: int, device: str) -> float:
    total, count = 0.0, 0
    for text in texts:
        tokens = tokenizer(text, add_special_tokens=False).input_ids[: max_length - 2]
        ids = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
        ids += [tokenizer.pad_token_id] * (max_length - len(ids))
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(x, labels=x.clone())
        n_valid = len(tokens) + 2
        total += float(out.loss) * n_valid
        count += n_valid
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="续训验收门：防遗忘 + 长文本改善")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from minimind_reborn import envs
    from minimind_reborn.models.config import MiniMindConfig
    from minimind_reborn.models.model import MiniMindForCausalLM
    from minimind_reborn.models.weights import load_inference_weights

    tokenizer = AutoTokenizer.from_pretrained(envs.tokenizer_path())
    device = args.device

    def load(path: str) -> MiniMindForCausalLM:
        m = MiniMindForCausalLM(MiniMindConfig()).eval().to(device)
        load_inference_weights(m, path, strict=True)
        return m

    old = load("out/archive/era1/pretrain_768.pth")
    new = load("out/pretrain_768.pth")  # 续训产物（save_weight=pretrain 覆盖同名）

    rng = random.Random(42)
    short_pool, long_pool = [], []
    with (envs.data_root() / "pretrain_t2t.jsonl").open("rb") as f:
        for line in f:
            if len(short_pool) >= 4000 and len(long_pool) >= 4000:
                break
            text = str(json.loads(line)["text"])
            if len(text) < 340 * 0.95 and len(short_pool) < 4000:
                short_pool.append(text)
            elif len(text) > 1024 and len(long_pool) < 4000:
                long_pool.append(text)
    shorts = rng.sample(short_pool, args.n)
    longs = rng.sample(long_pool, args.n)

    res = {}
    for name, texts, window in [("short_docs", shorts, 384), ("long_docs", longs, 1536)]:
        l_old = ce_loss(old, tokenizer, texts, window, device)
        l_new = ce_loss(new, tokenizer, texts, window, device)
        res[name] = {
            "old_base": round(l_old, 4),
            "new_base": round(l_new, 4),
            "delta_pct": round((l_new - l_old) / l_old * 100, 2),
        }
    res["gate"] = {
        "forgetting_ok": res["short_docs"]["delta_pct"] <= 5.0,
        "long_improved": res["long_docs"]["delta_pct"] < 0,
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if not all(res["gate"].values()):
        raise SystemExit("验收门未通过：防遗忘劣化 >5% 或长文本无改善——按回退线处置（lr 降 2e-5 或窗口 1024）")


if __name__ == "__main__":
    main()

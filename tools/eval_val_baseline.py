"""旧权重在新验证集（哈希分桶口径）上的基线评估（重训计划任务 2.1）。

对比三组：旧尾切 val（历史口径 1.956 的来源）、新哈希 val、新哈希 val 上
管线修复后的编码口径（轮次截断 + 残段不计损）。产出 JSON 到 stdout。
用法：uv run python tools/eval_val_baseline.py --weight out/archive/era1/full_sft_768.pth
"""

from __future__ import annotations

import argparse
import json

import torch

REPO_WEIGHT_DEFAULT = "out/archive/era1/full_sft_768.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description="旧权重 × 新验证集基线")
    parser.add_argument("--weight", default=REPO_WEIGHT_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=200, help="每口径最多评估的 batch 数")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from minimind_reborn import envs
    from minimind_reborn.data.datasets import PretrainDataset, SFTDataset  # noqa: F401
    from minimind_reborn.data.registry import file_path, resolve_dataset
    from minimind_reborn.models.config import MiniMindConfig
    from minimind_reborn.models.model import MiniMindForCausalLM
    from minimind_reborn.models.weights import load_inference_weights

    tokenizer = AutoTokenizer.from_pretrained(envs.tokenizer_path())
    model = MiniMindForCausalLM(MiniMindConfig()).eval().to(args.device)
    load_inference_weights(model, args.weight, strict=True)

    path = file_path(resolve_dataset("sft_t2t"))
    # 口径 A：新哈希分桶 val（C4 去重后），训练管线编码（C1 轮次截断 + C2 残段不计损）
    ds_new = SFTDataset(path, tokenizer, max_length=1536, eval_ratio=0.003, add_system_ratio=0.0)
    ev_new = ds_new.eval_view()
    assert ev_new is not None

    @torch.no_grad()
    def avg_loss(view, max_batches: int) -> tuple[float, int]:
        total, count, batches = 0.0, 0, 0
        for i in range(0, len(view), args.batch):
            if batches >= max_batches:
                break
            rows = [view[j] for j in range(i, min(i + args.batch, len(view)))]
            input_ids = torch.stack([r["input_ids"] for r in rows]).to(args.device)
            labels = torch.stack([r["labels"] for r in rows]).to(args.device)
            out = model(input_ids, labels=labels)
            n_tok = sum(int((r["labels"] != -100).sum()) for r in rows)
            total += float(out.loss) * n_tok
            count += n_tok
            batches += 1
        return total / max(count, 1), count

    loss_new, n_new = avg_loss(ev_new, args.max_batches)

    # 口径 B：旧尾切 val（历史口径对照）——临时用旧语义近似不可行，改用「同一数据集尾部切片」
    from minimind_reborn.data.datasets import JsonlIndexedDataset  # noqa: F401

    n_total = len(ds_new._offsets)
    tail_rows = list(range(int(n_total * 0.997), n_total))

    class _TailView(torch.utils.data.Dataset):
        def __init__(self, base, rows):
            self.base, self.rows = base, rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.base.encode(self.rows[i])

    loss_old_view, n_old = avg_loss(_TailView(ds_new, tail_rows), args.max_batches)

    print(
        json.dumps(
            {
                "weight": args.weight,
                "val_new_hash_loss_token_avg": round(loss_new, 4),
                "val_new_tokens": n_new,
                "val_old_tail_loss_token_avg": round(loss_old_view, 4),
                "val_old_tokens": n_old,
                "note": "token 加权平均 CE；新口径=哈希分桶+去重+轮次截断编码；旧口径=文件尾部 0.3% 切片同编码",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

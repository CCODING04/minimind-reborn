"""tokenizer 训练工具（官方 trainer/train_tokenizer.py 的移植，含空词表 bug 修复）。

背景：本项目默认采用官方已训好的 6400 词表 tokenizer（assets/tokenizer/，保证与官方
权重/数据互通）。此工具保留"自训 tokenizer"的能力——但换词表意味着历史权重/数据全部
不可复用，须重训全流程（官方 README 同样不建议轻易重训）。

用法：uv run python tools/train_tokenizer.py --data <jsonl> --out <dir> [--vocab-size 6400]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS_NUM = 36
VOCAB_SIZE = 6400


def get_texts(data_path: str | Path, max_lines: int = 0) -> Iterator[str]:
    """逐行产出用于训练 BPE 的文本，兼容两种主线数据格式（text / conversations）。

    官方修复过的坑：原先只解析 conversations，指向 pretrain 语料时会静默产出
    没有任何 merge 的空词表——这里对空产出显式 raise。
    """
    used = 0
    with open(data_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if max_lines and used >= max_lines:
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "text" in data:
                text = str(data["text"])
            else:
                contents = [item.get("content") for item in data.get("conversations", []) if item.get("content")]
                text = "\n".join(contents) if contents else ""
            if text.strip():
                used += 1
                yield text
    if used == 0:
        raise ValueError(f'{data_path} 中没有可用文本：每行应为 {{"text": ...}} 或 conversations 结构')


def train_tokenizer(data_path: str | Path, tokenizer_dir: str | Path, vocab_size: int = VOCAB_SIZE) -> None:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens_list = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|audio_start|>",
        "<|audio_end|>",
        "<|audio_pad|>",
        "<tts_pad>",
        "<tts_text_bos>",
        "<tts_text_eod>",
        "<tts_text_bos_single>",
    ]
    additional_tokens_list = [
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<think>",
        "</think>",
    ]
    buffer_tokens = [
        f"<|buffer{i}|>"
        for i in range(1, SPECIAL_TOKENS_NUM - len(special_tokens_list) - len(additional_tokens_list) + 1)
    ]
    all_special = special_tokens_list + additional_tokens_list + buffer_tokens

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=all_special,
    )
    tokenizer.train_from_iterator(get_texts(str(data_path)), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(special_tokens_list)

    out = Path(tokenizer_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / "tokenizer.json"))
    tokenizer.model.save(str(out))

    # added_tokens 里非 special 列表的标记转普通 token（与官方产物格式对齐）
    tok_json_path = out / "tokenizer.json"
    tok_data = json.loads(tok_json_path.read_text(encoding="utf-8"))
    for token_info in tok_data.get("added_tokens", []):
        if token_info["content"] not in special_tokens_list:
            token_info["special"] = False
    tok_json_path.write_text(json.dumps(tok_data, ensure_ascii=False, indent=2), encoding="utf-8")

    added_decoder = {
        str(tokenizer.token_to_id(t)): {
            "content": t,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": t in special_tokens_list,
        }
        for t in all_special
    }
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_decoder,
        "additional_special_tokens": [t for t in special_tokens_list if t != "<|endoftext|>"],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 131072,
        "pad_token": "<|endoftext|>",
        "spaces_between_special_tokens": False,
        "unk_token": "<|endoftext|>",
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    (out / "tokenizer_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tokenizer 训练完成：{out}（词表 {tokenizer.get_vocab_size()}）")


def main() -> None:
    parser = argparse.ArgumentParser(description="自训 tokenizer（默认建议使用官方 assets/tokenizer）")
    parser.add_argument("--data", required=True, help="训练语料 jsonl（pretrain 或 sft 格式均可）")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = parser.parse_args()

    train_tokenizer(args.data, args.out, vocab_size=args.vocab_size)


if __name__ == "__main__":
    main()

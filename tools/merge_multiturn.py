"""外部多轮数据清洗转换 + 与 sft_t2t 合并（重训计划档二 D1 对策）。

输入：tools/fetch_multiturn.py 下载的 Belle multiturn_chat_0.8M（jsonl，
instruction 含 Human:/Assistant: 轮次标记、output 为最终回复）。
输出：{data_root}/sft_t2t_mt.jsonl = sft_t2t 全量原样 + 外部转换行（流式拼接，
顺序无关——manifest 的哈希分桶与去重与顺序无关）。

过滤红线（重训计划 §数据工程）：
- 解析成功且 user/assistant 严格交替（Belle 标记切分失败即弃）
- 消息数 ≥3（真多轮）、assistant 回复非空
- 中文占比 ≥50%（用户侧字符）
- 总字符 ≤12000（字符级预过滤，token 级超窗交给训练侧 C1 轮次适配）
- 去污染：Suite A/C 评测探针命中的行剔除（评测泄露检查）
- 上限 --max-external 条（默认 30 万，确定性 shuffle seed 42 后截取）

用法：uv run python tools/merge_multiturn.py [--max-external 300000]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from pathlib import Path

from minimind_reborn import envs
from minimind_reborn.data.manifest import normalize_text

TURN_RE = re.compile(r"(?:^|\n)(Human:|Assistant:)")
ZH_RE = re.compile(r"[\u4e00-\u9fff]")
MAX_TOTAL_CHARS = 12000


def eval_probes() -> list[str]:
    """从 eval_multiturn.py 取 Suite A/C 的全部探针文本（单一事实源，import 而非复制）。"""
    spec = importlib.util.spec_from_file_location(
        "eval_multiturn", Path(__file__).parent / "eval_multiturn.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # noqa: PLC0415  仅取常量，不执行 main
    probes: list[str] = []
    for case in getattr(mod, "A_CASES", []):
        probes.extend(case.get("script", []))
        probes.extend(case.get("recall_keywords", []))
    probes.extend(getattr(mod, "C_QUESTIONS", []))
    return [p for p in probes if p]


class ProbeChecker:
    """评测探针命中检查：归一化后子串匹配（保守：宁错杀不泄露）。"""

    def __init__(self) -> None:
        self._needles = [normalize_text(p) for p in eval_probes() if len(normalize_text(p)) >= 4]

    def hits(self, text: str) -> bool:
        t = normalize_text(text)
        return any(n in t for n in self._needles)


def parse_belle_instruction(instruction: str) -> list[dict[str, str]] | None:
    """Belle instruction → 消息列表（不含最终回复）；结构不合法返回 None。

    instruction 形如 "Human:...\nAssistant:...\n...\nHuman:...\nAssistant:"——
    尾部的空 Assistant: 是约定式「待回复位」，其回答在 output 字段（非解析错误）。
    合法结构必须以 user 结尾（悬空待答），此前 user/assistant 严格交替。
    """
    parts = TURN_RE.split(instruction)
    # split 结果：[前导, 标记, 内容, 标记, 内容, ...]
    if len(parts) < 3 or parts[0].strip():
        return None
    messages: list[dict[str, str]] = []
    for i in range(1, len(parts) - 1, 2):
        role, content = parts[i], parts[i + 1].strip()
        if role == "Human:":
            if not content or (messages and messages[-1]["role"] == "user"):
                return None
            messages.append({"role": "user", "content": content})
        else:  # Assistant:
            if content:
                if messages and messages[-1]["role"] == "assistant":
                    return None
                messages.append({"role": "assistant", "content": content})
            else:
                # 空回复位只允许出现在末尾（悬空待答）；出现在中间即结构损坏
                if i != len(parts) - 2:
                    return None
    if not messages or messages[-1]["role"] != "user":
        return None
    return messages


def convert_record(rec: dict) -> list[dict[str, str]] | None:
    """Belle 样本 → conversations 消息列表；过滤红线不过返回 None。"""
    instruction, output = str(rec.get("instruction", "")), str(rec.get("output", "")).strip()
    if not output:
        return None
    messages = parse_belle_instruction(instruction)
    if messages is None:
        return None
    messages.append({"role": "assistant", "content": output})
    if len(messages) < 3:
        return None
    if any(not m["content"] for m in messages):
        return None
    if sum(1 for m in messages if m["role"] != "assistant") < 2:
        return None
    total_chars = sum(len(m["content"]) for m in messages)
    if total_chars > MAX_TOTAL_CHARS:
        return None
    zh = sum(len(ZH_RE.findall(m["content"])) for m in messages if m["role"] == "user")
    tot = sum(len(m["content"]) for m in messages if m["role"] == "user")
    if tot == 0 or zh / tot < 0.5:
        return None
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Belle 多轮数据清洗合并")
    parser.add_argument("--max-external", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = envs.data_root() / "raw" / "multiturn_chat_0.8M.json"
    base = envs.data_root() / "sft_t2t.jsonl"
    out = envs.data_root() / "sft_t2t_mt.jsonl"
    if not raw.exists():
        raise SystemExit(f"原始文件不存在：{raw}（先跑 tools/fetch_multiturn.py）")

    checker = ProbeChecker()
    stats = {"parsed": 0, "bad_parse": 0, "short": 0, "long": 0, "non_zh": 0, "probe_hit": 0, "kept": 0}
    external: list[dict] = []
    with raw.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                messages = convert_record(rec)
            except json.JSONDecodeError:
                messages = None
                stats["bad_parse"] += 1
                continue
            stats["parsed"] += 1
            if messages is None:
                # 归因粗分（保持脚本简单：长/非中/短序检查）
                text = str(rec.get("instruction", ""))
                if len(text) + len(str(rec.get("output", ""))) > MAX_TOTAL_CHARS:
                    stats["long"] += 1
                elif not text or len(ZH_RE.findall(text)) < 0.5 * len(text):
                    stats["non_zh"] += 1
                else:
                    stats["short"] += 1
                continue
            conv_text = json.dumps(messages, ensure_ascii=False)
            if checker.hits(conv_text):
                stats["probe_hit"] += 1
                continue
            stats["kept"] += 1
            external.append({"conversations": messages})

    rng = random.Random(args.seed)
    rng.shuffle(external)
    selected = external[: args.max_external]

    print(
        json.dumps(
            {
                "belle_lines": stats["parsed"] + stats["bad_parse"],
                **stats,
                "selected": len(selected),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # 合并：sft_t2t 流式原样 + 外部追加（顺序无关：manifest 哈希分桶/去重与顺序无关）
    tmp = out.with_suffix(".jsonl.tmp")
    n_base = 0
    with tmp.open("wb") as w, base.open("rb") as r:
        for line in r:
            if line.strip():
                w.write(line if line.endswith(b"\n") else line + b"\n")
                n_base += 1
        for rec in selected:
            w.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
    tmp.rename(out)
    print(f"合并完成：{out}（基础 {n_base} 行 + 外部 {len(selected)} 行）")


if __name__ == "__main__":
    main()

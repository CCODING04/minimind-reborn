"""collator：变长结构批处理（定长 tensor 字段走 torch 默认 collate，无需自定义）。"""

from __future__ import annotations

from typing import Any


def agent_collate(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """AgentRLDataset 的字段是变长 list，按列组装。"""
    return {
        "messages": [item["messages"] for item in batch],
        "tools": [item["tools"] for item in batch],
        "gt": [item["gt"] for item in batch],
    }

"""注册表测试：声明式资源未注册名必须报错并指明 json 路径。"""

from __future__ import annotations

import pytest

from minimind_reborn.data.registry import resolve_dataset


def test_unknown_name_raises_with_hint():
    with pytest.raises(ValueError, match="datasets.json"):
        resolve_dataset("no_such_dataset")


def test_known_names_resolve():
    for name in ("pretrain_t2t_mini", "sft_t2t_mini", "dpo_mini"):
        entry = resolve_dataset(name)
        assert entry.file.endswith(".jsonl") and entry.kind

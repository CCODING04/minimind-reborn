"""双后端（jsonl + tensorboard）同录正确性测试。

背景：metrics 曾出现 step 尺度混乱与交错写入——本测试锁定"jsonl 兜底 + tensorboard
可选后端同时启用"时两条通道的记录一致性与文件形态（事件文件须在 tensorboard/ 子目录）。
"""

from __future__ import annotations

import json

import pytest

from minimind_reborn.metrics import MetricLogger, build_backends


def _tb_installed() -> bool:
    try:
        import tensorboard  # noqa: F401

        return True
    except ImportError:
        return False


tensorboard_required = pytest.mark.skipif(
    not _tb_installed(), reason="tensorboard 未安装（uv sync --extra metrics）"
)


@tensorboard_required
def test_dual_backend_jsonl_and_tensorboard(tmp_path):
    run_dir = tmp_path / "runs" / "demo" / "20260924_000000"
    backends = build_backends(["tensorboard"], run_dir)
    # jsonl 兜底恒在首位 + tensorboard 可选后端
    assert [type(b).__name__ for b in backends] == ["JSONLBackend", "TensorBoardBackend"]
    ml = MetricLogger(backends)

    expect: dict[str, dict[int, float]] = {"train/loss": {}, "val/loss": {}}
    for step in range(1, 11):
        mapping = {"train/loss": 7.0 - step * 0.1, "val/loss": 7.2 - step * 0.1}
        for k, v in mapping.items():
            expect[k][step] = v
        ml.log(mapping, step)
    ml.close()

    # ① jsonl 行数与内容精确对齐
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 20
    for entry in lines:
        assert entry["value"] == expect[entry["key"]][entry["step"]]

    # ② 事件文件在 tensorboard/ 子目录（不混入四件套）
    tb_dir = run_dir / "tensorboard"
    assert list(tb_dir.glob("events.out.tfevents.*")), "TB 事件文件未生成"

    # ③ EventAccumulator 读回与写入一致（数值无损）
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(tb_dir))
    acc.Reload()
    for key in expect:
        got = {e.step: e.value for e in acc.Scalars(key)}
        for step, want in expect[key].items():
            assert abs(got[step] - want) < 1e-6, f"TB 记录偏差 {key}@{step}"


@tensorboard_required
def test_tensorboard_events_in_subdir_only(tmp_path):
    """事件文件必须落在 tensorboard/ 子目录，run_dir 根不允许出现 tfevents。"""
    run_dir = tmp_path / "run"
    backends = build_backends(["tensorboard"], run_dir)
    ml = MetricLogger(backends)
    ml.log({"train/loss": 1.0}, 1)
    ml.close()
    root_files = [p.name for p in run_dir.iterdir() if p.is_file()]
    assert not any("tfevents" in n for n in root_files), "事件文件混入 run_dir 根"
    assert (run_dir / "tensorboard").exists()

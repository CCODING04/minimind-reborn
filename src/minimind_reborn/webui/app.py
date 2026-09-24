"""Web 试验台后端（Phase 1：F1 对话测试 + F2 阶段对比）。

分层纪律（llm-inference §1）：本层只 import 推理门面 MiniMindLLM，不碰模型内部；
请求可追溯（request_id/延迟/停止原因）随 SSE 事件下发。

实现说明：
- SSE 生成器是同步生成器——Starlette 会把它放进线程池执行，
  因此生成线程的 queue.get() 阻塞不会卡事件循环；
- 本文件刻意不用 `from __future__ import annotations`：请求模型（ChatRequest）
  定义在 create_app 闭包内，字符串化注解会让 FastAPI 解析不到它。
"""

import json
import time
import uuid
from pathlib import Path
from queue import Queue
from threading import Thread

from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.inference.chat import render_chat
from minimind_reborn.loggers import get_logger
from minimind_reborn.webui.pool import EnginePool
from minimind_reborn.webui.registry import ModelEntry, discover

logger = get_logger("webui")

_ASSETS = Path(__file__).resolve().parent / "assets"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


class QueueStreamer:
    """把逐 token 文本流接进队列（TextStreamer 回调转发）。"""

    def __init__(self, tokenizer, queue: Queue):
        from transformers import TextStreamer

        self.queue = queue
        self.inner = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.inner.on_finalized_text = self.on_finalized_text  # type: ignore[method-assign]

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)

    def put(self, item):
        self.inner.put(item)

    def end(self):
        self.inner.end()


def create_app(device: str | None = None, dtype: str = "bf16"):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError as e:
        raise SystemExit(f"缺少服务依赖（{e}）。请安装：uv sync --extra serving") from None

    from transformers import AutoTokenizer

    from minimind_reborn import envs
    from minimind_reborn.utils.io import read_json

    app = FastAPI(title="minimind_reborn 试验台", version="phase1")
    pool = EnginePool(device=device, dtype=dtype)
    AutoTokenizer.from_pretrained(envs.tokenizer_path())  # 预热 tokenizer 缓存；QueueStreamer 内部各自加载

    class GenParams(BaseModel):
        temperature: float = 0.7
        top_p: float = 0.9
        top_k: int = 50
        repetition_penalty: float = 1.0
        max_new_tokens: int = 256

    class ChatRequest(BaseModel):
        model_ids: list[str]  # 1 个 = 对话模式；≥2 个 = 对比模式
        use_best: bool = False
        messages: list[dict] = Field(default_factory=list)  # 完整对话历史（末位是本次 user 提问）
        pretrain_mode: bool = False  # base 模型直拼 bos+text，不走 chat template
        open_thinking: bool = False
        gen: GenParams = Field(default_factory=GenParams)

    def _entries() -> dict[str, ModelEntry]:
        return {e.model_id: e for e in discover()}

    @app.get("/")
    async def index():
        return FileResponse(_ASSETS / "index.html")

    @app.get("/echarts.min.js")
    async def echarts_js():
        return FileResponse(_ASSETS / "echarts.min.js", media_type="application/javascript")

    @app.get("/api/models")
    async def list_models():
        return [
            {
                "model_id": e.model_id,
                "stage": e.stage,
                "recipe_name": e.recipe_name,
                "trained_at": e.trained_at,
                "has_best": e.best_path is not None,
                "hidden_size": e.hidden_size,
                "use_moe": e.use_moe,
            }
            for e in _entries().values()
        ]

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "loaded": pool.current_id}

    # ============ Phase 2：训练观测（F3/F4，只读 runs/ 产物） ============
    @app.get("/api/runs")
    async def list_runs():
        """运行注册表：runs/recipe/ts 三层目录的元信息清单。"""
        runs = []
        for cfg_path in sorted((envs.run_root()).glob("*/*/config.json")):
            try:
                snap = read_json(cfg_path)
            except (OSError, json.JSONDecodeError):
                continue
            run_dir = cfg_path.parent
            best_val = None
            metrics_file = run_dir / "metrics.jsonl"
            if metrics_file.exists():
                for line in metrics_file.open(encoding="utf-8"):
                    d = json.loads(line)
                    if d.get("key") == "val/loss":
                        best_val = d["value"] if best_val is None else min(best_val, d["value"])
            runs.append(
                {
                    "run_id": f"{cfg_path.parent.parent.name}/{run_dir.name}",
                    "recipe_name": cfg_path.parent.parent.name,
                    "trained_at": run_dir.name,
                    "stage": snap.get("stage"),
                    "best_val_loss": best_val,
                    "has_metrics": metrics_file.exists(),
                    "has_log": (run_dir / "train.log").exists(),
                }
            )
        return sorted(runs, key=lambda r: r["trained_at"], reverse=True)

    @app.get("/api/runs/{recipe}/{trained_at}/metrics")
    async def run_metrics(recipe: str, trained_at: str):
        """metrics.jsonl → 按 key 分组的时间序列（供 ECharts 折线）。"""
        path = envs.run_root() / recipe / trained_at / "metrics.jsonl"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"metrics.jsonl 不存在：{recipe}/{trained_at}")
        series: dict[str, list[list[float]]] = {}
        for line in path.open(encoding="utf-8"):
            d = json.loads(line)
            if d["value"] != d["value"]:  # 过滤 NaN
                continue
            series.setdefault(d["key"], []).append([d["step"], round(d["value"], 6)])
        return {"run_id": f"{recipe}/{trained_at}", "series": [{"key": k, "points": p} for k, p in series.items()]}

    @app.get("/api/runs/{recipe}/{trained_at}/log")
    async def run_log(recipe: str, trained_at: str, tail: int = 200):
        path = envs.run_root() / recipe / trained_at / "train.log"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"train.log 不存在：{recipe}/{trained_at}")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"run_id": f"{recipe}/{trained_at}", "total": len(lines), "lines": lines[-max(tail, 1) :]}

    @app.get("/api/runs/{recipe}/{trained_at}/meta")
    async def run_meta(recipe: str, trained_at: str):
        """配置快照 + 环境指纹摘要（可追溯四件套的另外两件）。"""
        run_dir = envs.run_root() / recipe / trained_at
        cfg = read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else None
        env = read_json(run_dir / "env.json") if (run_dir / "env.json").exists() else None
        return {"run_id": f"{recipe}/{trained_at}", "config": cfg, "env": env}

    def _stream_one(
        entry: ModelEntry,
        use_best: bool,
        messages: list[dict],
        gen: GenParams,
        pretrain_mode: bool,
        open_thinking: bool,
        tag: str,
    ):
        """单个模型的流式生成：加载 → 渲染 → 逐 token 产出 SSE 事件（同步生成器）。"""
        request_id = uuid.uuid4().hex[:12]
        yield _sse({"type": "model_start", "model_id": tag, "request_id": request_id})
        started = time.perf_counter()
        try:
            llm = pool.get(entry, use_best=use_best)
        except Exception as e:  # noqa: BLE001
            logger.error("模型加载失败 %s: %s", tag, e)
            yield _sse({"type": "error", "model_id": tag, "message": f"模型加载失败：{e}"})
            return

        if pretrain_mode:
            prompt_text = llm.tokenizer.bos_token + messages[-1]["content"]
        else:
            prompt_text = render_chat(llm.tokenizer, messages, open_thinking=open_thinking)
        inputs = llm.tokenizer(prompt_text, return_tensors="pt", truncation=True).to(llm.device)

        queue: Queue = Queue()
        streamer = QueueStreamer(llm.tokenizer, queue)

        def _generate():
            try:
                llm.generate_tokens(
                    inputs["input_ids"],
                    GenerateConfig(
                        temperature=gen.temperature,
                        top_p=gen.top_p,
                        top_k=gen.top_k,
                        repetition_penalty=gen.repetition_penalty,
                        max_new_tokens=gen.max_new_tokens,
                    ),
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=gen.max_new_tokens,
                    streamer=streamer,
                )
            except Exception as e:  # noqa: BLE001
                logger.error("生成失败 request_id=%s: %s", request_id, e)
                queue.put({"__error__": str(e)})
                queue.put(None)

        Thread(target=_generate, daemon=True).start()

        emitted = ""
        while True:
            chunk = queue.get()
            if chunk is None:
                break
            if isinstance(chunk, dict):
                yield _sse({"type": "error", "model_id": tag, "message": chunk.get("__error__", "unknown")})
                continue
            emitted += chunk
            yield _sse({"type": "delta", "model_id": tag, "text": chunk})

        # 停止三重之 3：截断尾部 eos 残留后，以下发的最终文本为准
        ids = llm.tokenizer(emitted, add_special_tokens=False).input_ids
        eos = llm.tokenizer.eos_token_id
        if eos is not None and eos in ids:
            ids = ids[: ids.index(eos)]
        clean = llm.tokenizer.decode(ids, skip_special_tokens=True)
        latency = time.perf_counter() - started
        yield _sse(
            {
                "type": "done",
                "model_id": tag,
                "request_id": request_id,
                "content": clean,
                "latency_ms": round(latency * 1000, 1),
                "tokens_per_s": round(len(ids) / max(latency, 1e-6), 1),
            }
        )

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        entries = _entries()
        unknown = [m for m in req.model_ids if m not in entries]
        if unknown:
            raise HTTPException(status_code=404, detail=f"未知模型 {unknown}；可选项：{sorted(entries)}")
        if not req.messages or req.messages[-1].get("role") != "user":
            raise HTTPException(status_code=422, detail="messages 末位必须是本次 user 提问")

        def event_stream():  # 同步生成器：Starlette 放线程池，queue.get 阻塞不卡事件循环
            yield _sse({"type": "start", "models": req.model_ids})
            # 按需加载策略：多模型对比 = 顺序加载-生成-卸载
            for model_id in req.model_ids:
                yield from _stream_one(
                    entries[model_id],
                    req.use_best,
                    req.messages,
                    req.gen,
                    req.pretrain_mode,
                    req.open_thinking,
                    model_id,
                )
            yield _sse({"type": "end"})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app

"""OpenAI 风格推理服务（官方 serve_openai_api.py 的重构）。

/v1/chat/completions：流式（reasoning_content 分流）与非流式；tool_calls 解析。
可选依赖 fail-fast：fastapi/uvicorn 缺失时在启动期报错（python-style §2）。
"""
from __future__ import annotations

import argparse
import json
import re
import time
from queue import Queue
from threading import Thread

from minimind_reborn.configuration.schemas import GenerateConfig
from minimind_reborn.inference.chat import split_reasoning
from minimind_reborn.loggers import catch_main, get_logger

logger = get_logger("serve")


def parse_response(text: str) -> tuple[str, str | None, list[dict] | None]:
    """拆 <think> 与 <tool_call>：返回 (content, reasoning, tool_calls)。"""
    reasoning, content = split_reasoning(text)
    tool_calls = []
    for i, m in enumerate(re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            tool_calls.append({
                "id": f"call_{int(time.time())}_{i}", "type": "function",
                "function": {"name": call.get("name", ""),
                             "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)},
            })
        except json.JSONDecodeError:
            pass
    if tool_calls:
        content = re.sub(r"<tool_call>.*?</tool_call>", "", content or "", flags=re.DOTALL).strip()
    return content, reasoning, tool_calls or None


@catch_main
def main() -> None:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
        import uvicorn
    except ImportError as e:
        raise SystemExit(f"缺少服务依赖（{e}）。请安装 serving extras：uv sync --extra serving") from None

    from transformers import TextStreamer

    from minimind_reborn.inference.engine import MiniMindLLM

    parser = argparse.ArgumentParser(description="minimind_reborn OpenAI 风格服务")
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    llm = MiniMindLLM.build(args.weight, hidden_size=args.hidden_size,
                            num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe, device=args.device)
    app = FastAPI(title="minimind_reborn")

    class ChatRequest(BaseModel):
        model: str = "minimind-reborn"
        messages: list
        temperature: float = 0.7
        top_p: float = 0.92
        max_tokens: int = 1024
        stream: bool = True
        tools: list = Field(default_factory=list)
        open_thinking: bool = False

    class _Streamer(TextStreamer):
        def __init__(self, tokenizer, queue: Queue):
            super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
            self.queue = queue

        def on_finalized_text(self, text: str, stream_end: bool = False):
            self.queue.put(text)
            if stream_end:
                self.queue.put(None)

    def stream_chunks(request: ChatRequest):
        prompt = llm.tokenizer.apply_chat_template(request.messages, tokenize=False,
                                                   add_generation_prompt=True,
                                                   tools=request.tools or None,
                                                   open_thinking=request.open_thinking)
        inputs = llm.tokenizer(prompt, return_tensors="pt", truncation=True).to(llm.device)
        queue: Queue = Queue()
        streamer = _Streamer(llm.tokenizer, queue)

        def _generate():
            try:
                llm.generate_tokens(inputs["input_ids"], GenerateConfig(temperature=request.temperature,
                                                                        top_p=request.top_p),
                                    attention_mask=inputs["attention_mask"],
                                    max_new_tokens=request.max_tokens, streamer=streamer)
            except Exception as e:  # noqa: BLE001
                queue.put(json.dumps({"error": str(e)}, ensure_ascii=False))
                queue.put(None)

        Thread(target=_generate, daemon=True).start()
        emitted = 0
        full_text = ""
        thinking_ended = not request.open_thinking
        while True:
            chunk = queue.get()
            if chunk is None:
                break
            if isinstance(chunk, str) and chunk.startswith('{"error"'):
                yield chunk
                continue
            full_text += chunk
            if not thinking_ended:
                pos = full_text.find("</think>")
                if pos >= 0:
                    thinking_ended = True
                    new_reason = full_text[emitted:pos]
                    if new_reason:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_reason}}]}, ensure_ascii=False)
                    emitted = len(full_text)
                else:
                    new_reason = full_text[emitted:]
                    if new_reason:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_reason}}]}, ensure_ascii=False)
                    emitted = len(full_text)
            else:
                new_content = full_text[emitted:]
                if new_content:
                    yield json.dumps({"choices": [{"delta": {"content": new_content}}]}, ensure_ascii=False)
                emitted = len(full_text)
        content, _, tool_calls = parse_response(full_text)
        if tool_calls:
            yield json.dumps({"choices": [{"delta": {"tool_calls": tool_calls}}]}, ensure_ascii=False)
        yield json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}]},
                         ensure_ascii=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        if request.stream:
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in stream_chunks(request)),
                media_type="text/event-stream",
            )
        result = llm.generate_text(request.messages,
                                   GenerateConfig(temperature=request.temperature, top_p=request.top_p),
                                   tools=request.tools or None, open_thinking=request.open_thinking)
        content, reasoning, tool_calls = parse_response(result["content"])
        message: dict = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "id": f"chatcmpl-{result['request_id']}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if tool_calls else result["finish_reason"]}],
        }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    logger.info("服务启动 http://%s:%d/v1/chat/completions", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

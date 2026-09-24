"""Web 试验台薄入口（project-structure §2.2：只做 解析参数 → 组装 → 启动）。

用法：uv run python tools/webui.py --port 8000
"""

from __future__ import annotations

import argparse

from minimind_reborn.loggers import catch_main, get_logger

logger = get_logger("webui")


@catch_main
def main() -> None:
    parser = argparse.ArgumentParser(description="minimind_reborn Web 试验台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="推理设备（默认自动）")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    args = parser.parse_args()

    import uvicorn

    from minimind_reborn.webui.app import create_app

    app = create_app(device=args.device, dtype=args.dtype)
    logger.info("试验台启动 http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

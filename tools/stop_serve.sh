#!/bin/bash
# minimind_reborn serve.py 停止脚本
# 用法: bash tools/stop_serve.sh [--port 8899]
# 流程: SIGTERM → 等待 10s → SIGKILL → GPU 显存核实

PORT=8899
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# ① 按 ss 找端口监听者（最可靠）
PID=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1)
if [ -z "$PID" ]; then
  # ② 回退 pgrep（[.] 技巧 + 排除自身 PID 防误杀调用器 shell）
  PID=$(pgrep -f "tools/serve[.]py" | grep -v "^$$\$" | head -1)
fi

if [ -z "$PID" ]; then
  echo "serve.py 无运行进程，无需停止"
  exit 0
fi

CMD=$(ps -o cmd= -p "$PID" 2>/dev/null | head -c 80)
echo "停止 serve.py (pid=$PID): $CMD"

kill "$PID" 2>/dev/null
for _ in $(seq 1 10); do
  kill -0 "$PID" 2>/dev/null || break
  sleep 1
done
if kill -0 "$PID" 2>/dev/null; then
  echo "SIGTERM 未生效，SIGKILL"
  kill -9 "$PID" 2>/dev/null
  sleep 1
fi

if kill -0 "$PID" 2>/dev/null; then
  echo "停止失败（pid=$PID 仍存活）"; exit 1
fi
echo "✓ serve.py 已停止（pid=$PID）"

# GPU 显存核实
if command -v nvidia-smi &>/dev/null; then
  sleep 2
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | head -4
fi

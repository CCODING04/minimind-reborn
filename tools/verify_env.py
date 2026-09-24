"""环境验证脚本（env-dependencies §2：装完 torch 必跑）。"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="torch/CUDA 环境验证")
    parser.add_argument("--allow-cpu", action="store_true", help="允许无 GPU 环境（CI 用）")
    args = parser.parse_args()

    import torch

    print("torch:", torch.__version__, "| cuda runtime:", torch.version.cuda)
    print("GPU 可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("设备数:", torch.cuda.device_count())
        print("设备名:", torch.cuda.get_device_name(0))
    elif not args.allow_cpu:
        raise SystemExit(
            "torch.cuda.is_available()=False：极可能装到了 CPU wheel 或驱动不匹配。"
            "请查 nvidia-smi 的驱动 CUDA 版本后按 skills/ml-dev-spec env-dependencies §2 重装"
        )
    x = torch.randn(1024, 1024, device="cuda" if torch.cuda.is_available() else "cpu")
    print("矩阵乘 OK:", (x @ x).sum().item() == (x @ x).sum().item())


if __name__ == "__main__":
    main()

"""
start_isat_sam.py — ISAT-SAM Backend 启动脚本
─────────────────────────────────────────────
启动 ISAT-SAM Backend 服务

使用方式:
    python scripts/start_isat_sam.py
    python scripts/start_isat_sam.py --model sam_vit_b --port 8000
"""

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def start_isat_sam_server(model="mobile_sam", host="127.0.0.1", port=8000, open_browser=True):
    """启动 ISAT-SAM Backend 服务"""

    print("=" * 60)
    print("  ISAT-SAM Backend 服务")
    print("=" * 60)
    print(f"  SAM 模型: {model}")
    print(f"  服务地址: http://{host}:{port}")
    print("=" * 60)

    # 检查 isat-sam-backend 是否安装
    try:
        subprocess.run(["isat-sam-backend", "--help"],
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n[ERROR] isat-sam-backend 未安装")
        print("\n请先运行安装脚本:")
        print("  python scripts/setup_isat_sam.py")
        return

    # 启动服务
    cmd = [
        "isat-sam-backend",
        "--checkpoint", model,
        "--host", host,
        "--port", str(port)
    ]

    print("\n[INFO] 启动服务中...")
    print(f"[INFO] 命令: {' '.join(cmd)}")
    print("\n按 Ctrl+C 停止服务\n")

    try:
        # 打开浏览器
        if open_browser:
            time.sleep(2)
            webbrowser.open(f"http://{host}:{port}")

        # 运行服务
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n[INFO] 服务已停止")


def main():
    parser = argparse.ArgumentParser(description="ISAT-SAM Backend 服务启动脚本")
    parser.add_argument(
        "--model",
        type=str,
        default="mobile_sam",
        choices=["mobile_sam", "sam_vit_b", "sam_vit_l", "sam_vit_h"],
        help="SAM 模型类型（默认: mobile_sam）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="服务地址（默认: 127.0.0.1）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="端口（默认: 8000）"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="不自动打开浏览器"
    )

    args = parser.parse_args()

    start_isat_sam_server(
        model=args.model,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser
    )


if __name__ == "__main__":
    main()

"""
start_isat_sam.py — ISAT-SAM 启动脚本
──────────────────────────────────────
启动 ISAT-SAM 标注工具

使用方式:
    python scripts/start_isat_sam.py
"""

import subprocess
import sys
from pathlib import Path


def start_isat_sam():
    """启动 ISAT-SAM"""

    print("=" * 60)
    print("  ISAT-SAM 标注工具")
    print("=" * 60)

    # 检查 isat-sam 是否安装
    try:
        result = subprocess.run(
            ["isat-sam", "--help" if False else "--version"],
            capture_output=True,
            text=True
        )
    except FileNotFoundError:
        print("\n[ERROR] isat-sam 未安装")
        print("\n请先运行安装脚本:")
        print("  python scripts/setup_isat_sam.py")
        return

    print("\n[INFO] 启动 ISAT-SAM...")
    print("\n按 Ctrl+C 停止\n")

    try:
        # 直接运行 isat-sam
        subprocess.run(["isat-sam"])
    except KeyboardInterrupt:
        print("\n\n[INFO] 已停止 ISAT-SAM")


def main():
    start_isat_sam()


if __name__ == "__main__":
    main()

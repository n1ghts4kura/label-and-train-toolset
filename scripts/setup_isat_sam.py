"""
setup_isat_sam.py — ISAT-SAM 环境安装脚本
──────────────────────────────────────────
安装 ISAT-SAM 标注工具及其依赖

使用方式:
    python scripts/setup_isat_sam.py
    python scripts/setup_isat_sam.py --sam-model mobile_sam
"""

import argparse
import subprocess
import sys
from pathlib import Path


def install_isat_sam(sam_model="mobile_sam"):
    """安装 ISAT-SAM 及其依赖"""
    print("=" * 60)
    print("  ISAT-SAM 环境安装")
    print("=" * 60)

    # 检查 Python 版本
    print("\n[1/4] 检查 Python 版本...")
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8+ required")
        return False
    print(f"    Python {sys.version_info.major}.{sys.version_info.minor} ✓")

    # 创建 conda 环境（如果使用 conda）
    use_conda = Path("C:/Users/18252/anaconda3/Scripts/conda.exe").exists() or \
                Path("C:/ProgramData/anaconda3/Scripts/conda.exe").exists()

    if use_conda:
        print("\n[2/4] 创建 conda 环境 isat_sam...")
        subprocess.run(["conda", "create", "-n", "isat_sam", "python=3.8", "-y"],
                       capture_output=True)
        print("    conda 环境创建完成")
        print("    激活方式: conda activate isat_sam")
    else:
        print("\n[2/4] 创建虚拟环境 isat_sam...")
        subprocess.run([sys.executable, "-m", "venv", "venv_isat_sam"],
                       capture_output=True)
        print("    虚拟环境创建完成")
        print("    激活方式: venv_isat_sam\\Scripts\\activate.bat")

    # 确定 pip 路径
    if use_conda:
        pip_cmd = ["conda", "run", "-n", "isat_sam", "pip"]
    else:
        pip_cmd = ["./venv_isat_sam/Scripts/python.exe"]

    # 安装 isat-sam-backend
    print("\n[3/4] 安装 isat-sam-backend...")
    result = subprocess.run(pip_cmd + ["install", "isat-sam-backend"])
    if result.returncode != 0:
        print("    [WARN] pip 安装失败，尝试使用 conda...")
        if use_conda:
            subprocess.run(["conda", "install", "-n", "isat_sam", "-c", "pip", "isat-sam-backend", "-y"],
                         capture_output=True)

    # 下载 SAM 模型
    print(f"\n[4/4] 下载 SAM 模型: {sam_model}...")
    subprocess.run(pip_cmd + [
        "-m", "isat_sam_backend", "model", "--download", sam_model
    ], capture_output=True)

    print("\n" + "=" * 60)
    print("  ISAT-SAM 安装完成！")
    print("=" * 60)
    print("\n启动方式:")
    if use_conda:
        print("  1. conda activate isat_sam")
    else:
        print("  1. venv_isat_sam\\Scripts\\activate.bat")
    print("  2. isat-sam-backend")
    print("  3. 运行 ISAT 工具连接 localhost:8000")
    print("\n或者下载 ISAT GUI: https://github.com/yatengLG/ISAT/releases")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="ISAT-SAM 环境安装脚本")
    parser.add_argument(
        "--sam-model",
        type=str,
        default="mobile_sam",
        choices=["mobile_sam", "sam_vit_b", "sam_vit_l", "sam_vit_h"],
        help="SAM 模型类型（默认: mobile_sam）"
    )
    args = parser.parse_args()

    install_isat_sam(args.sam_model)


if __name__ == "__main__":
    main()

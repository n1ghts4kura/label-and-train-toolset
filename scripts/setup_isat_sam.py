"""
setup_isat_sam.py — ISAT-SAM 环境安装脚本
──────────────────────────────────────────
安装 ISAT-SAM 标注工具及其依赖

使用方式:
    python scripts/setup_isat_sam.py
"""

import subprocess
import sys
from pathlib import Path


def install_isat_sam():
    """安装 ISAT-SAM 及其依赖"""
    print("=" * 60)
    print("  ISAT-SAM 环境安装")
    print("=" * 60)

    # 检查 Python 版本
    print("\n[1/3] 检查 Python 版本...")
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8+ required")
        return False
    print(f"    Python {sys.version_info.major}.{sys.version_info.minor} ✓")

    # 创建 conda 环境（如果使用 conda）
    use_conda = Path("C:/Users/18252/anaconda3/Scripts/conda.exe").exists() or \
                Path("C:/ProgramData/anaconda3/Scripts/conda.exe").exists()

    if use_conda:
        print("\n[2/3] 创建 conda 环境 isat_sam...")
        result = subprocess.run(
            ["conda", "create", "-n", "isat_sam", "python=3.10", "-y"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("    [WARN] conda 环境创建可能失败，尝试继续...")
        print("    激活方式: conda activate isat_sam")
    else:
        print("\n[2/3] 创建虚拟环境 venv_isat_sam...")
        subprocess.run([sys.executable, "-m", "venv", "venv_isat_sam"],
                       capture_output=True)
        print("    激活方式: venv_isat_sam\\Scripts\\activate.bat")

    # 确定 pip 路径
    if use_conda:
        pip_cmd = ["conda", "run", "-n", "isat_sam", "pip"]
    else:
        pip_cmd = ["./venv_isat_sam/Scripts/python.exe"]

    # 安装 isat-sam
    print("\n[3/3] 安装 isat-sam...")
    result = subprocess.run(pip_cmd + ["install", "isat-sam"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [ERROR] 安装失败: {result.stderr}")
        return False

    print("    isat-sam 安装成功!")

    print("\n" + "=" * 60)
    print("  ISAT-SAM 安装完成！")
    print("=" * 60)
    print("\n启动方式:")
    if use_conda:
        print("  1. conda activate isat_sam")
    else:
        print("  1. venv_isat_sam\\Scripts\\activate.bat")
    print("  2. isat-sam")
    print("\n或直接运行:")
    if use_conda:
        print("  conda activate isat_sam && isat-sam")
    else:
        print("  venv_isat_sam\\Scripts\\activate.bat && isat-sam")
    print("=" * 60)

    return True


def main():
    install_isat_sam()


if __name__ == "__main__":
    main()

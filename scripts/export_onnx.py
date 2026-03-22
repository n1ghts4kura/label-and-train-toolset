"""
export_onnx.py
────────────────────────────────────────────────────────────────
将训练好的 YOLOv5-seg .pt 权重导出为 ONNX 格式

使用方式:
    python export_onnx.py                       # 自动查找最新 best.pt
    python export_onnx.py --weights path/to/best.pt  # 指定权重路径

输出:
    与输入 .pt 同目录下的 best.onnx 文件
"""

import argparse
import subprocess
import sys
from glob import glob
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

# 路径常量
CONFIG_YAML   = PROJECT_ROOT / "configs" / "train_config.yaml"
YOLOV5_ROOT  = PROJECT_ROOT / "yolov5"
EXPORT_SCRIPT = YOLOV5_ROOT / "export.py"


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_latest_best_pt(project: str, name: str) -> Path | None:
    """在 yolov5/<project>/<name>*/weights/best.pt 中找编号最大的"""
    base = YOLOV5_ROOT / project
    if not base.exists():
        return None

    pattern = str(base / f"{name}*" / "weights" / "best.pt")
    candidates = glob(pattern)
    if not candidates:
        return None

    def sort_key(p):
        dir_name = Path(p).parent.parent.name
        suffix = dir_name.replace(name, "")
        nums = [int(x) for x in suffix.split() if x.isdigit()]
        return nums[0] if nums else 0

    return Path(sorted(candidates, key=sort_key)[-1])


def main():
    parser = argparse.ArgumentParser(description="YOLOv5-seg ONNX 导出")
    parser.add_argument(
        "--weights", type=str, default=None,
        help="指定 .pt 权重路径（默认自动查找最新 best.pt）"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="导出时的推理图片尺寸（默认 640）"
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset 版本（默认 17）"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" export_onnx.py — YOLOv5-seg 导出为 ONNX")
    print("=" * 60)

    if not EXPORT_SCRIPT.exists():
        print(f"\n[ERROR] 导出脚本不存在: {EXPORT_SCRIPT}")
        print("请先运行 setup_env.bat 克隆 yolov5 仓库")
        sys.exit(1)

    # 确定权重路径
    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.is_absolute():
            weights_path = PROJECT_ROOT / weights_path
    else:
        cfg = load_config(CONFIG_YAML)
        project = cfg.get("project", "runs/train-seg")
        name    = cfg.get("name", "exp")
        weights_path = find_latest_best_pt(project, name)

        if weights_path is None:
            print(f"\n[ERROR] 未找到 best.pt，请先完成训练（python scripts/train.py）")
            print(f"        或使用 --weights 参数手动指定权重路径")
            sys.exit(1)

    if not weights_path.exists():
        print(f"\n[ERROR] 权重文件不存在: {weights_path}")
        sys.exit(1)

    print(f"\n[INFO] 输入权重: {weights_path}")
    expected_onnx = weights_path.with_suffix(".onnx")
    print(f"[INFO] 预期输出: {expected_onnx}")
    print(f"[INFO] 图片尺寸: {args.imgsz}x{args.imgsz}")
    print(f"[INFO] ONNX opset: {args.opset}")

    # 构建导出命令
    cmd = [
        sys.executable, str(EXPORT_SCRIPT),
        "--weights",  str(weights_path),
        "--include",  "onnx",
        "--simplify",
        "--opset",    str(args.opset),
        "--imgsz",    str(args.imgsz), str(args.imgsz),
        "--device",   "cpu",
    ]

    print(f"\n[CMD] {' '.join(cmd)}\n")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=str(YOLOV5_ROOT))

    print("\n" + "=" * 60)
    if result.returncode == 0:
        if expected_onnx.exists():
            size_mb = expected_onnx.stat().st_size / (1024 * 1024)
            print(f"[DONE] 导出成功！")
            print(f"[INFO] ONNX 模型路径: {expected_onnx}")
            print(f"[INFO] 文件大小: {size_mb:.2f} MB")
            print(f"\n[INFO] 测试模型: python scripts/test.py <图片路径> --weights {expected_onnx}")
        else:
            print(f"[WARN] 导出命令成功但未找到输出文件: {expected_onnx}")
    else:
        print(f"[ERROR] 导出失败，返回码: {result.returncode}")
        print("[HINT] 请确认已安装 onnx 和 onnxslim: pip install onnx onnxslim")
    print("=" * 60)


if __name__ == "__main__":
    main()

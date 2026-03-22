"""
train.py
────────────────────────────────────────────────────────────────
YOLOv5-seg 训练入口脚本

使用方式:
    python train.py              # 自动检测断点，按需 resume
    python train.py --fresh      # 强制从头开始，忽略 last.pt

功能:
    - 读取 configs/train_config.yaml 中的训练参数
    - 自动检测最新 last.pt，断点续训
    - 调用 yolov5/segment/train.py 完成实际训练
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
CONFIG_YAML  = PROJECT_ROOT / "configs" / "train_config.yaml"
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"
YOLOV5_ROOT  = PROJECT_ROOT / "yolov5"
TRAIN_SCRIPT = YOLOV5_ROOT / "segment" / "train.py"


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_latest_last_pt(project: str, name: str) -> Path | None:
    """在 yolov5/<project>/<name>*/weights/last.pt 中找最新的 last.pt"""
    base = YOLOV5_ROOT / project
    if not base.exists():
        return None

    pattern = str(base / f"{name}*" / "weights" / "last.pt")
    candidates = glob(pattern)
    if not candidates:
        return None

    def sort_key(p):
        dir_name = Path(p).parent.parent.name
        nums = [int(x) for x in dir_name.replace(name, "").split() if x.isdigit()]
        return nums[0] if nums else 0

    return Path(sorted(candidates, key=sort_key)[-1])


def validate_prerequisites():
    """检查必要文件是否存在"""
    errors = []

    if not YOLOV5_ROOT.exists():
        errors.append(
            f"  ✗ yolov5/ 目录不存在。请先运行 setup_env.bat 克隆仓库，\n"
            f"    或手动执行: git clone https://github.com/ultralytics/yolov5.git yolov5"
        )

    if not TRAIN_SCRIPT.exists():
        errors.append(f"  ✗ 训练脚本不存在: {TRAIN_SCRIPT}")

    if not CONFIG_YAML.exists():
        errors.append(f"  ✗ 配置文件不存在: {CONFIG_YAML}")

    if DATASET_YAML.exists():
        with open(DATASET_YAML, "r", encoding="utf-8") as f:
            ds = yaml.safe_load(f)
        if ds and ds.get("path") == "TO_BE_SET_BY_SPLIT_SCRIPT":
            errors.append(
                "  ✗ dataset.yaml 的 path 尚未设置，请先运行: python scripts/split_dataset.py"
            )
    else:
        errors.append(f"  ✗ 数据集配置不存在: {DATASET_YAML}")

    if errors:
        print("\n[ERROR] 前置检查失败:\n")
        for e in errors:
            print(e)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="YOLOv5-seg 训练入口")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="强制从头开始训练，忽略已有的 last.pt"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" train.py — YOLOv5-seg 微调训练")
    print("=" * 60)

    # 前置检查
    validate_prerequisites()

    # 读取配置
    cfg = load_config(CONFIG_YAML)
    print(f"\n[INFO] 已加载配置: {CONFIG_YAML}")

    # Windows 下 workers 钳制到 ≤ 2
    workers = min(int(cfg.get("workers", 2)), 2)

    # 检测是否需要 resume
    project = cfg.get("project", "runs/train-seg")
    name    = cfg.get("name", "exp")

    last_pt = None if args.fresh else find_latest_last_pt(project, name)

    if last_pt:
        print(f"[INFO] 检测到断点权重: {last_pt}")
        print("[INFO] 将以 --resume 模式继续训练（如需从头开始请加 --fresh 参数）")
    else:
        print("[INFO] 未检测到断点权重，从头开始训练")

    # 构建命令
    if last_pt:
        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--resume", str(last_pt),
        ]
    else:
        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--weights",    str(PROJECT_ROOT / cfg["weights"]),
            "--data",       str(DATASET_YAML),
            "--epochs",     str(cfg.get("epochs", 100)),
            "--batch-size", str(cfg.get("batch_size", 8)),
            "--imgsz",      str(cfg.get("imgsz", 640)),
            "--device",     str(cfg.get("device", "0")),
            "--optimizer",  str(cfg.get("optimizer", "SGD")),
            "--patience",   str(cfg.get("patience", 50)),
            "--save-period",str(cfg.get("save_period", -1)),
            "--workers",    str(workers),
            "--project",    project,
            "--name",       name,
        ]

    print(f"\n[CMD] {' '.join(cmd)}\n")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=str(YOLOV5_ROOT))

    print("\n" + "=" * 60)
    if result.returncode == 0:
        runs_dir = PROJECT_ROOT / project
        print("[DONE] 训练完成！")
        best_candidates = sorted(
            (YOLOV5_ROOT / project).glob(f"{name}*/weights/best.pt")
        )
        if best_candidates:
            print(f"[INFO] 最佳权重: {best_candidates[-1]}")
        print("[INFO] 下一步运行: python scripts/export_onnx.py")
    else:
        print(f"[ERROR] 训练异常退出，返回码: {result.returncode}")
        print("[INFO] 下次运行 python scripts/train.py 将自动从 last.pt 恢复训练")
    print("=" * 60)


if __name__ == "__main__":
    main()

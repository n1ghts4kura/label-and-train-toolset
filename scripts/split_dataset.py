"""
split_dataset.py
────────────────────────────────────────────────────────────────
数据集划分脚本

将 train_data/ 目录下的图片按 train : val 比例划分。

使用方式:
    python split_dataset.py                # 使用默认/当前项目配置
    python split_dataset.py --project xxx # 指定项目配置

功能:
    1. 扫描 train_data/ 中的图片和标签配对
    2. 按比例划分训练集和验证集
    3. 复制到 images/train, images/val, labels/train, labels/val
    4. 自动生成 dataset.yaml
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_manager import get_config
from core.dataset_splitter import split_dataset


def main():
    parser = argparse.ArgumentParser(
        description="数据集划分 (train:val = N:1)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="项目配置名称（查找 configs/projects/{name}.yaml）"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" split_dataset.py — 数据集划分")
    print("=" * 60)

    # 加载配置
    cfg = get_config()

    if args.project:
        project_file = PROJECT_ROOT / "configs" / "projects" / f"{args.project}.yaml"
        if not project_file.exists():
            print(f"[ERROR] 项目配置文件不存在: {project_file}")
            sys.exit(1)
        cfg.switch_project_by_path(project_file)

    print(f"\n[INFO] 项目名称: {cfg.project_name}")
    print(f"[INFO] 数据目录: {cfg.data_root}")
    print(f"[INFO] 划分比例: {cfg.train_val_ratio}:1 (train:val)")

    # 检查目录
    if not cfg.data_root.exists():
        print(f"\n[ERROR] 目录不存在: {cfg.data_root}")
        print("请先运行: python scripts/export_labels.py")
        sys.exit(1)

    # 执行划分
    result = split_dataset(
        data_root=cfg.data_root,
        output_root=cfg.data_root,
        ratio=cfg.train_val_ratio,
        seed=cfg.random_seed
    )

    # 输出结果
    print(f"\n[INFO] 总配对数: {result['total_pairs']}")
    print(f"[INFO] 训练集: {result['train_count']} ({result['train_ratio']:.1f}%)")
    print(f"[INFO] 验证集: {result['val_count']} ({result['val_ratio']:.1f}%)")

    if result['missing_labels']:
        print(f"\n[WARN] 无标签图片数: {len(result['missing_labels'])}")

    if result['warnings']:
        for w in result['warnings']:
            print(f"       ! {w}")

    # 生成 dataset.yaml
    dataset_yaml = cfg.save_dataset_yaml()
    print(f"\n[INFO] dataset.yaml 已生成: {dataset_yaml}")

    print("\n[DONE] 数据集划分完成！")
    print("       下一步运行: python scripts/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

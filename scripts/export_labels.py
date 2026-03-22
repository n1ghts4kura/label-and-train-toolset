"""
export_labels.py
────────────────────────────────────────────────────────────────
ISAT JSON → YOLO 分割标签转换脚本

使用方式:
    python export_labels.py                    # 使用默认/当前项目配置
    python export_labels.py --project my_proj  # 指定项目配置
    python export_labels.py --check           # 仅检查，不执行写入

功能:
    1. 解析 ISAT JSON 标注文件
    2. 将坐标转换为 YOLO 分割格式（归一化）
    3. 复制图片和标签到 train_data/
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_manager import get_config
from core.isat_converter import convert_isat_to_yolo


def main():
    parser = argparse.ArgumentParser(
        description="ISAT JSON → YOLO 分割标签转换"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="项目配置名称（查找 configs/projects/{name}.yaml）"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅检查，不执行任何写入操作"
    )
    parser.add_argument(
        "--include-crowd",
        action="store_true",
        help="保留 iscrowd=true 的标注（默认跳过）"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" export_labels.py — ISAT → YOLO 分割标签转换")
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
    print(f"[INFO] 原始目录: {cfg.origin_dir}")
    print(f"[INFO] 输出目录: {cfg.data_root}")
    print(f"[INFO] 类别映射: {cfg.class_map}")

    # 检查目录
    if not cfg.origin_dir.exists():
        print(f"\n[ERROR] 目录不存在: {cfg.origin_dir}")
        print("请先创建目录并放入 ISAT 标注文件和图片")
        sys.exit(1)

    # 执行转换
    result = convert_isat_to_yolo(
        origin_dir=cfg.origin_dir,
        output_dir=cfg.data_root,
        isat_yaml=cfg.isat_yaml,
        include_crowd=args.include_crowd,
        dry_run=args.check
    )

    # 输出结果
    print(f"\n[INFO] ISAT JSON 文件数: {result['total_jsons']}")
    print(f"[INFO] 处理图片数: {result['processed_images']}")
    print(f"[INFO] 总目标数: {result['total_objects']}")

    if result["warnings"]:
        print(f"\n[WARN] 警告信息 ({len(result['warnings'])} 条):")
        for w in result["warnings"][:10]:
            print(f"       ! {w}")

    if result["errors"]:
        print(f"\n[ERROR] 处理错误 ({len(result['errors'])} 条):")
        for e in result["errors"]:
            print(f"       ✗ {e}")

    if args.check:
        print("\n[CHECK] 仅检查模式，不执行任何写入操作。")
        print("        移除 --check 参数后重新运行以完成导出。")
    else:
        print(f"\n[DONE] 转换完成！")
        print(f"       下一步运行: python scripts/split_dataset.py")

    print("=" * 60)


if __name__ == "__main__":
    main()

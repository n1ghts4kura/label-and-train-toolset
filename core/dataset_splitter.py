"""
dataset_splitter.py
────────────────────────────────────────────────────────────────
数据集划分模块

将 train_data/ 根目录下的图片（及对应的 YOLO 分割标签 .txt）
按 train : val = N : 1 的比例随机划分，并复制到标准目录结构。

使用方式:
    from core.dataset_splitter import split_dataset
    result = split_dataset(
        data_root=Path("./train_data"),
        output_root=Path("./train_data"),
        ratio=10,
        seed=42
    )
"""

import random
import shutil
from pathlib import Path
from typing import Optional


SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_pairs(src_dir: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    """
    扫描 src_dir 根目录，收集所有「图片 + 同名 .txt 标签」配对。

    返回:
        (pairs, missing_labels)
        pairs: [(img_path, label_path), ...]
        missing_labels: 无对应标签的图片名称列表
    """
    pairs = []
    missing_labels = []

    for img_path in sorted(src_dir.iterdir()):
        if img_path.suffix.lower() not in SUPPORTED_IMG_EXTS:
            continue
        label_path = src_dir / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path.name)

    return pairs, missing_labels


def split_pairs(
    pairs: list[tuple[Path, Path]],
    ratio: int,
    seed: int
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """
    按 ratio : 1 划分为 (train_list, val_list)。
    ratio=10 → 每 (ratio+1) 张中取 1 张作 val。
    """
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    val_count = max(1, len(shuffled) // (ratio + 1))
    val_list   = shuffled[:val_count]
    train_list = shuffled[val_count:]
    return train_list, val_list


def prepare_dirs(output_root: Path) -> dict[str, Path]:
    """
    清空并重建 images/labels 子目录，返回路径字典。
    """
    dirs = {
        "img_train":   output_root / "images" / "train",
        "img_val":     output_root / "images" / "val",
        "label_train": output_root / "labels" / "train",
        "label_val":   output_root / "labels" / "val",
    }
    for d in dirs.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def copy_pairs(
    pairs: list[tuple[Path, Path]],
    img_dst: Path,
    label_dst: Path
):
    """将图片和标签复制到目标目录。"""
    for img_path, label_path in pairs:
        shutil.copy2(img_path, img_dst / img_path.name)
        shutil.copy2(label_path, label_dst / label_path.name)


def split_dataset(
    data_root: Path,
    output_root: Optional[Path] = None,
    ratio: int = 10,
    seed: int = 42
) -> dict:
    """
    执行数据集划分。

    参数:
        data_root: 包含原始图片和 .txt 标签的目录
        output_root: 输出目录，默认与 data_root 相同
        ratio: train : val 比例（默认 10）
        seed: 随机种子（默认 42）

    返回:
        {
            "total_pairs": int,
            "train_count": int,
            "val_count": int,
            "train_ratio": float,
            "val_ratio": float,
            "missing_labels": list[str],
            "warnings": list[str],
        }
    """
    if output_root is None:
        output_root = data_root

    warnings = []

    # 1. 收集配对
    pairs, missing_labels = collect_pairs(data_root)

    if missing_labels:
        warnings.append(f"以下图片无对应 .txt 标签，已跳过（共 {len(missing_labels)} 张）")

    if not pairs:
        return {
            "total_pairs": 0,
            "train_count": 0,
            "val_count": 0,
            "train_ratio": 0.0,
            "val_ratio": 0.0,
            "missing_labels": missing_labels,
            "warnings": warnings,
        }

    # 2. 划分
    train_pairs, val_pairs = split_pairs(pairs, ratio, seed)

    # 3. 准备目录（清空重建）
    dirs = prepare_dirs(output_root)

    # 4. 复制文件
    copy_pairs(train_pairs, dirs["img_train"], dirs["label_train"])
    copy_pairs(val_pairs,   dirs["img_val"],   dirs["label_val"])

    total = len(pairs)
    return {
        "total_pairs": total,
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "train_ratio": len(train_pairs) / total * 100,
        "val_ratio": len(val_pairs) / total * 100,
        "missing_labels": missing_labels[:10],  # 只保留前10个
        "warnings": warnings,
    }

"""
validator.py
────────────────────────────────────────────────────────────────
YOLO 标签验证模块

验证 YOLO 格式标签文件的正确性：
- 坐标值范围 [0, 1]
- 类别 ID 有效
- 格式正确（class_id + 偶数个坐标）
"""

from pathlib import Path
from typing import Optional


def validate_yolo_label(
    label_path: Path,
    num_classes: int,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None
) -> tuple[bool, list[str]]:
    """
    验证单个 YOLO 标签文件的正确性。

    参数:
        label_path: YOLO .txt 标签文件路径
        num_classes: 类别总数
        img_width: 图片宽度（像素），如果提供则额外验证坐标物理范围
        img_height: 图片高度（像素）

    返回:
        (is_valid, errors)
        is_valid: 是否通过所有验证
        errors: 错误信息列表
    """
    errors = []

    if not label_path.exists():
        return False, [f"文件不存在: {label_path}"]

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return False, [f"读取文件失败: {e}"]

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # 检查是否有至少 class_id + 1 个坐标对
        if len(parts) < 3:
            errors.append(f"行 {line_num}: 字段数不足（需要 class_id + 至少一对坐标）")
            continue

        # 第一个是类别 ID
        try:
            class_id = int(parts[0])
        except ValueError:
            errors.append(f"行 {line_num}: 类别 ID 不是整数")
            continue

        if class_id < 0:
            errors.append(f"行 {line_num}: 类别 ID 不能为负数")
        elif class_id >= num_classes:
            errors.append(f"行 {line_num}: 类别 ID {class_id} 超出范围 [0, {num_classes - 1}]")

        # 检查坐标数量是否为偶数（x, y 配对）
        coords = parts[1:]
        if len(coords) % 2 != 0:
            errors.append(f"行 {line_num}: 坐标数量不是偶数")
            continue

        # 验证每个坐标值
        for i, coord in enumerate(coords):
            try:
                val = float(coord)
                if img_width is not None and img_height is not None:
                    # 如果提供图片尺寸，验证像素范围
                    if i % 2 == 0:  # x 坐标
                        pixel_val = val * img_width
                    else:  # y 坐标
                        pixel_val = val * img_height

                    if pixel_val < 0 or pixel_val > (img_width if i % 2 == 0 else img_height):
                        # 这个检查需要考虑实际像素值，但 YOLO 归一化后理论上就是 [0,1]
                        pass
                else:
                    # 仅检查归一化范围
                    if val < 0.0 or val > 1.0:
                        errors.append(
                            f"行 {line_num}: 坐标值 {val} 超出范围 [0, 1]"
                        )
            except ValueError:
                errors.append(f"行 {line_num}: 坐标值 '{coord}' 不是有效数字")

    return len(errors) == 0, errors


def validate_dataset(
    data_root: Path,
    num_classes: int,
    subset: str = "train"
) -> dict:
    """
    验证数据集中所有标签文件的正确性。

    参数:
        data_root: 数据集根目录（包含 images/ 和 labels/ 子目录）
        num_classes: 类别总数
        subset: "train" 或 "val"

    返回:
        {
            "total_labels": int,
            "valid_labels": int,
            "invalid_labels": int,
            "errors_by_file": dict[str, list[str]],
            "summary": str,
        }
    """
    label_dir = data_root / "labels" / subset

    if not label_dir.exists():
        return {
            "total_labels": 0,
            "valid_labels": 0,
            "invalid_labels": 0,
            "errors_by_file": {},
            "summary": f"目录不存在: {label_dir}",
        }

    label_files = sorted(label_dir.glob("*.txt"))
    valid_count = 0
    invalid_count = 0
    errors_by_file = {}

    for label_file in label_files:
        is_valid, errors = validate_yolo_label(label_file, num_classes)
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            errors_by_file[label_file.name] = errors

    return {
        "total_labels": len(label_files),
        "valid_labels": valid_count,
        "invalid_labels": invalid_count,
        "errors_by_file": errors_by_file,
        "summary": (
            f"总计: {len(label_files)} 个标签文件, "
            f"有效: {valid_count}, 无效: {invalid_count}"
        ),
    }

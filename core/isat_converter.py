"""
isat_converter.py
────────────────────────────────────────────────────────────────
ISAT JSON → YOLO 分割标签转换模块

从 ISAT-SAM 导出的 JSON 标注文件转换为 YOLO 分割格式。

ISAT JSON → YOLO txt 转换规则:
    输入: ISAT JSON，segmentation 为像素坐标列表 [[x1,y1], [x2,y2], ...]
    输出: YOLO txt，每行格式为：
          <class_id> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ...
          坐标已归一化到 [0, 1]（相对图片宽高）
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import yaml


SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_class_map(isat_yaml: Path) -> dict[str, int]:
    """
    从 isat.yaml 读取类别顺序，返回 {category_name: class_id}。
    __background__ 始终跳过（不参与 YOLO 类别编号）。
    """
    if not isat_yaml.exists():
        raise FileNotFoundError(
            f"未找到类别配置文件: {isat_yaml}\n"
            f"请确保 isat.yaml 存在"
        )

    with open(isat_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    labels = cfg.get("label", [])
    class_map = {}
    idx = 0
    for label in labels:
        name = label.get("name", "")
        if name and name != "__background__":
            class_map[name] = idx
            idx += 1

    return class_map


def parse_isat_json(
    json_path: Path,
    class_map: dict,
    include_crowd: bool = False
) -> tuple[str, list[str], list[str]]:
    """
    解析单个 ISAT JSON 文件，返回 YOLO 格式的行列表。

    参数:
        json_path: ISAT JSON 文件路径
        class_map: 类别映射 {category_name: class_id}
        include_crowd: 是否保留 iscrowd=true 的标注

    返回:
        (image_filename, yolo_lines, warnings)
        yolo_lines: List[str]，每行为一个目标的 YOLO 分割标注
        warnings:   List[str]，解析过程中的警告信息
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})
    img_name = info.get("name", "")
    width    = info.get("width", 0)
    height   = info.get("height", 0)

    if not img_name:
        img_name = json_path.stem  # 退而用 json 文件名

    if width == 0 or height == 0:
        return img_name, [], [f"{json_path.name}: 图片宽高为 0，无法归一化坐标"]

    yolo_lines = []
    warnings   = []

    for obj in data.get("objects", []):
        category = obj.get("category", "")
        iscrowd  = obj.get("iscrowd", False)
        seg      = obj.get("segmentation", [])

        # 跳过 crowd 对象（可选保留）
        if iscrowd and not include_crowd:
            continue

        # 跳过未知类别
        if category not in class_map:
            if category != "__background__":
                warnings.append(
                    f"{json_path.name} [{category}]: 类别不在 isat.yaml 中，已跳过"
                )
            continue

        # 跳过空 segmentation
        if not seg or len(seg) < 3:
            warnings.append(
                f"{json_path.name} [{category}]: segmentation 点数 < 3，已跳过"
            )
            continue

        class_id = class_map[category]

        # 归一化坐标：pixel [x,y] → [x/w, y/h]
        norm_coords = []
        valid = True
        for point in seg:
            if len(point) < 2:
                valid = False
                break
            x_norm = point[0] / width
            y_norm = point[1] / height
            # 钳制到 [0, 1]，防止轻微超出边界的标注破坏格式
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            norm_coords.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])

        if not valid:
            warnings.append(f"{json_path.name} [{category}]: 点格式异常，已跳过")
            continue

        yolo_lines.append(f"{class_id} " + " ".join(norm_coords))

    return img_name, yolo_lines, warnings


def convert_isat_to_yolo(
    origin_dir: Path,
    output_dir: Path,
    isat_yaml: Path,
    include_crowd: bool = False,
    dry_run: bool = False
) -> dict:
    """
    将 origin_dir 中的 ISAT JSON 标注文件转换为 YOLO 分割格式，
    并复制图片和标签到 output_dir。

    参数:
        origin_dir: 包含原始图片和 ISAT JSON 的目录
        output_dir: 输出目录（YOLO 图片和标签）
        isat_yaml: ISAT 类别配置文件路径
        include_crowd: 是否保留 iscrowd=true 的标注
        dry_run: True 则仅检查不执行任何写入操作

    返回:
        {
            "total_jsons": int,
            "processed_images": int,
            "total_objects": int,
            "warnings": list[str],
            "errors": list[str],
        }
    """
    # 1. 加载类别映射
    try:
        class_map = load_class_map(isat_yaml)
    except FileNotFoundError as e:
        return {
            "total_jsons": 0,
            "processed_images": 0,
            "total_objects": 0,
            "warnings": [],
            "errors": [str(e)],
        }

    # 2. 扫描 ISAT JSON 文件
    json_files = sorted(origin_dir.glob("*.json"))

    isat_jsons = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                d = json.load(f)
            if d.get("info", {}).get("description") == "ISAT":
                isat_jsons.append(jf)
        except Exception:
            pass

    # 3. 解析并收集转换结果
    results   = []
    all_warns = []

    for jf in isat_jsons:
        img_name, yolo_lines, warns = parse_isat_json(
            jf, class_map, include_crowd=include_crowd
        )
        all_warns.extend(warns)

        # 查找对应图片
        img_path = None
        for ext in SUPPORTED_IMG_EXTS:
            candidate = origin_dir / (Path(img_name).stem + ext)
            if candidate.exists():
                img_path = candidate
                break
            # 也尝试 json 同名图片
            candidate2 = origin_dir / (jf.stem + ext)
            if candidate2.exists():
                img_path = candidate2
                break

        if img_path is None:
            all_warns.append(
                f"{jf.name}: 找不到对应图片（{img_name}），已跳过"
            )
            continue

        results.append((img_path, yolo_lines, jf))

    # 4. 执行转换（如果非 dry_run）
    errors = []
    copied_imgs = 0
    written_txts = 0

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

        for img_path, yolo_lines, json_path in results:
            try:
                # 复制图片
                dst_img = output_dir / img_path.name
                shutil.copy2(img_path, dst_img)
                copied_imgs += 1

                # 写入 YOLO txt（同名）
                dst_txt = output_dir / (img_path.stem + ".txt")
                with open(dst_txt, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))
                    if yolo_lines:
                        f.write("\n")
                written_txts += 1

            except Exception as e:
                errors.append(f"{img_path.name}: {e}")

    # 5. 统计无标注的图片
    no_label = [img.name for img, lines, _ in results if len(lines) == 0]
    all_warns.extend([f"无标注: {n}" for n in no_label])

    return {
        "total_jsons": len(isat_jsons),
        "processed_images": copied_imgs if not dry_run else len(results),
        "total_objects": sum(len(lines) for _, lines, _ in results),
        "warnings": all_warns,
        "errors": errors,
    }

"""
test.py — YOLOv5-seg ONNX 推理脚本
────────────────────────────────────────────────────────────────
使用方式:
    python test.py <图片路径>                          # 使用默认配置
    python test.py <图片路径> --weights best.onnx     # 指定模型
    python test.py <图片路径> --conf 0.4 --iou 0.5    # 自定义阈值

输出:
    output_test_validation.jpg
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.yolo_inference import YOLOInference
from core.config_manager import get_config


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv5-seg ONNX 推理 → output_test_validation.jpg"
    )
    parser.add_argument("image", type=str, help="测试图片路径")
    parser.add_argument(
        "--weights", type=str, default=None,
        help="ONNX 模型路径（默认: 项目根目录 best.onnx）"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="置信度阈值（默认 0.25）"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU 阈值（默认 0.45）"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="推理尺寸（默认 640）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出图片路径（默认: output_test_validation.jpg）"
    )
    args = parser.parse_args()

    print("=" * 62)
    print("  test.py — YOLOv5-seg ONNX 推理")
    print("=" * 62)

    # 确定路径
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    if args.weights:
        onnx_path = Path(args.weights)
        if not onnx_path.is_absolute():
            onnx_path = PROJECT_ROOT / onnx_path
    else:
        onnx_path = PROJECT_ROOT / "best.onnx"
        if not onnx_path.exists():
            candidates = sorted(PROJECT_ROOT.glob("*.onnx"))
            if candidates:
                onnx_path = candidates[0]

    # 获取类别信息
    cfg = get_config()
    class_names = [c["name"] for c in cfg.classes] if cfg.classes else ["unknown"]

    print(f"[INFO] 图片   : {image_path}")
    print(f"[INFO] 模型    : {onnx_path}")
    print(f"[INFO] 类别    : {class_names}")
    print(f"[INFO] 置信度  : {args.conf}  |  IoU : {args.iou}")

    if not image_path.exists():
        print(f"[ERROR] 图片不存在: {image_path}")
        sys.exit(1)
    if not onnx_path.exists():
        print(f"[ERROR] 模型不存在: {onnx_path}")
        print("请先运行: python scripts/export_onnx.py")
        sys.exit(1)

    # 推理
    infer = YOLOInference(
        str(onnx_path),
        class_names=class_names
    )

    output_path = args.output or str(PROJECT_ROOT / "output_test_validation.jpg")
    result = infer.detect_and_draw(
        str(image_path),
        output_path=output_path,
        conf_thres=args.conf,
        iou_thres=args.iou,
        imgsz=args.imgsz
    )

    print(f"\n[DONE] 输出已保存: {output_path}")
    print("=" * 62)

    # 用系统默认查看器打开
    try:
        import os
        os.startfile(output_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()

"""
test_video.py — YOLOv5-seg 视频流推理可视化
────────────────────────────────────────────────────────────────
使用方式:
    python scripts/test_video.py <mp4路径> --weights best.onnx
    python scripts/test_video.py <mp4路径> --weights best.onnx --no-mask --imgsz 320

控制:
    [q] - 退出程序
    [s] - 保存当前帧截图
    [空格] - 暂停/继续
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.yolo_inference import YOLOInference
from core.config_manager import get_config, ConfigManager


def draw_detections(frame: np.ndarray, boxes, masks, scores, cls_ids,
                    class_names, colors, conf_thres=0.25) -> np.ndarray:
    """在一帧上绘制检测结果（框 + mask + 标签）"""
    if len(boxes) == 0:
        return frame

    # 绘制半透明 mask
    overlay = frame.astype(np.float32)
    for idx, binary in enumerate(masks):
        color = colors[int(cls_ids[idx]) % len(colors)]
        for c, cv in enumerate(color):
            # binary 是 bool 或 0/1，需要确保形状匹配
            mask_3ch = np.stack([binary] * 3, axis=-1) if binary.ndim == 2 else binary
            overlay[:, :, c] = np.where(
                mask_3ch[:, :, c],
                overlay[:, :, c] * 0.55 + cv * 0.45,
                overlay[:, :, c]
            )
    overlay = overlay.clip(0, 255).astype(np.uint8)

    # 绘制边框 + 标签
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx].tolist()
        conf   = float(scores[idx])
        cls_id = int(cls_ids[idx])
        color  = colors[cls_id % len(colors)]
        name   = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        label  = f"{name} {conf:.2f}"

        # 边框（3px 描边）
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # 标签背景
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        by1 = max(int(y1) - th - 4, 0)
        cv2.rectangle(overlay, (int(x1), by1), (int(x1) + tw, by1 + th + 2), color, -1)
        cv2.putText(overlay, label, (int(x1), by1 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return overlay


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv5-seg 视频流推理 → 实时可视化"
    )
    parser.add_argument("video", type=str, help="MP4 视频路径")
    parser.add_argument(
        "--weights", type=str, default=None,
        help="ONNX 模型路径（默认: 项目根目录 best.onnx）"
    )
    parser.add_argument(
        "--conf", type=float, default=0.3,
        help="置信度阈值（默认 0.3）"
    )
    parser.add_argument(
        "--iou", type=float, default=0.4,
        help="NMS IoU 阈值（默认 0.4）"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="推理尺寸（默认 640）"
    )
    parser.add_argument(
        "--fps", type=int, default=0,
        help="显示帧率（0=跟随原始视频）"
    )
    parser.add_argument(
        "--no-mask", action="store_true",
        help="跳过掩码计算（大幅提升帧率，仅显示检测框）"
    )
    args = parser.parse_args()

    print("=" * 62)
    print("  test_video.py — YOLOv5-seg 视频流推理")
    print("=" * 62)

    # 路径解析
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

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

    # 加载项目配置
    ConfigManager.get_instance().switch_project("rmyc_sim_v1")
    cfg = get_config()
    class_names = [c["name"] for c in cfg.classes] if cfg.classes else ["unknown"]

    print(f"[INFO] 视频   : {video_path}")
    print(f"[INFO] 模型   : {onnx_path}")
    print(f"[INFO] 类别   : {class_names}")
    print(f"[INFO] 阈值   : conf={args.conf}  iou={args.iou}  imgsz={args.imgsz}")
    print(f"[INFO] Mask   : {'关闭' if args.no_mask else '开启'}")
    print(f"[INFO] 控制   : [q]退出  [s]保存截图  [空格]暂停")
    print("=" * 62)

    if not video_path.exists():
        print(f"[ERROR] 视频不存在: {video_path}")
        sys.exit(1)
    if not onnx_path.exists():
        print(f"[ERROR] 模型不存在: {onnx_path}")
        print("请先运行: python scripts/export_onnx.py")
        sys.exit(1)

    # 初始化推理器
    infer = YOLOInference(str(onnx_path), class_names=class_names)

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        sys.exit(1)

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win_name = f"YOLOv5-seg 推理 — {video_path.name}"

    print(f"[INFO] 视频信息: {orig_fps:.1f}fps, {total_frames} 帧")
    print("[INFO] 开始播放，按 [q] 退出， [s] 保存当前帧...")

    frame_idx = 0
    pause = False
    result_frame = None
    boxes = []

    while True:
        if not pause:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] 视频播放完毕")
                break
            frame_idx += 1

            # 推理
            boxes, masks, scores, cls_ids = infer.detect(
                frame, conf_thres=args.conf, iou_thres=args.iou,
                imgsz=args.imgsz, with_mask=not args.no_mask
            )

            # 绘制结果
            result_frame = draw_detections(
                frame, boxes, masks, scores, cls_ids,
                class_names, infer.colors, conf_thres=args.conf
            )
        else:
            if result_frame is None:
                ret, result_frame = cap.read()
                if not ret:
                    print("[INFO] 视频播放完毕")
                    break

        # 显示 FPS 信息
        display_fps = args.fps if args.fps > 0 else int(orig_fps)
        info_text = f"Frame: {frame_idx}/{total_frames}  |  FPS: ~{display_fps}  |  conf: {args.conf}"
        cv2.putText(result_frame, info_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # 标注检测数量
        det_text = f"Detections: {len(boxes)}"
        cv2.putText(result_frame, det_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow(win_name, result_frame)

        key = cv2.waitKey(1 if not pause else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = PROJECT_ROOT / f"video_screenshot_{frame_idx:06d}.jpg"
            cv2.imwrite(str(save_path), result_frame)
            print(f"[INFO] 截图已保存: {save_path}")
        elif key == ord(' '):  # 空格暂停
            pause = not pause

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] 程序结束")
    print("=" * 62)


if __name__ == "__main__":
    main()

"""
extract_frames.py — MP4 视频帧提取工具
────────────────────────────────────────
将 MP4 视频按指定帧率或间隔提取为 JPG 图片

使用方式:
    python scripts/extract_frames.py video.mp4
    python scripts/extract_frames.py video.mp4 --fps 30
    python scripts/extract_frames.py video.mp4 --interval 5  # 每5帧提取1张
    python scripts/extract_frames.py video.mp4 -o ./frames/
    python scripts/extract_frames.py video.mp4 --start 100 --end 500  # 提取指定区间
"""

import argparse
import cv2
import os
from pathlib import Path
from datetime import datetime


def extract_frames(video_path, output_dir=None, fps=None, interval=None,
                   start_frame=None, end_frame=None, quality=95):
    """
    从视频中提取帧为 JPG 图片

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录（默认: 视频文件名_frames/）
        fps: 指定帧率提取（如 30 表示每秒提取 30 张）
        interval: 帧间隔提取（如 5 表示每 5 帧提取 1 张）
        start_frame: 起始帧号
        end_frame: 结束帧号
        quality: JPG 质量 (1-100)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] 视频文件不存在: {video_path}")
        return

    # 默认输出目录
    if output_dir is None:
        output_dir = video_path.stem + "_frames"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    # 获取视频信息
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print("=" * 60)
    print(f"  视频帧提取工具")
    print("=" * 60)
    print(f"  视频文件: {video_path.name}")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {video_fps:.2f} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {duration:.2f} 秒")
    print("=" * 60)

    # 确定提取模式
    if fps is not None:
        mode = "指定帧率"
        frame_interval = int(video_fps / fps)
        if frame_interval < 1:
            frame_interval = 1
    elif interval is not None:
        mode = "帧间隔"
        frame_interval = int(interval)
    else:
        mode = "默认帧间隔"
        frame_interval = int(video_fps / 30) or 1  # 默认每秒30帧

    print(f"  提取模式: {mode}")
    print(f"  提取间隔: 每 {frame_interval} 帧提取 1 张")
    if start_frame is not None:
        print(f"  起始帧: {start_frame}")
    if end_frame is not None:
        print(f"  结束帧: {end_frame}")
    print(f"  JPG 质量: {quality}")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)

    # 提取帧
    frame_count = 0
    extracted_count = 0
    start_frame = start_frame or 0
    end_frame = end_frame or total_frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # 检查是否在范围内
        if current_frame > end_frame:
            break

        # 检查是否需要提取
        if (current_frame - start_frame) % frame_interval == 0:
            # 生成输出文件名（使用帧号而不是时间戳，保证唯一性）
            output_name = f"{video_path.stem}_{current_frame:06d}.jpg"
            output_path = output_dir / output_name

            # 保存帧
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            extracted_count += 1

            if extracted_count % 100 == 0:
                print(f"  已提取: {extracted_count} 张 (帧 {current_frame}/{total_frames})")

        frame_count += 1

    cap.release()

    print("=" * 60)
    print(f"  完成！共提取 {extracted_count} 张图片")
    print(f"  保存位置: {output_dir.absolute()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="MP4 视频帧提取为 JPG 图片"
    )
    parser.add_argument("video", type=str, help="视频文件路径 (MP4/AVI/MOV)")
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="输出目录（默认: 视频名_frames/）"
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="指定帧率提取（如 30 表示每秒提取 30 张）"
    )
    parser.add_argument(
        "--interval", type=int, default=None,
        help="帧间隔（如 5 表示每 5 帧提取 1 张）"
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="起始帧号"
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="结束帧号"
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPG 质量 1-100（默认 95）"
    )

    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        interval=args.interval,
        start_frame=args.start,
        end_frame=args.end,
        quality=args.quality
    )


if __name__ == "__main__":
    main()

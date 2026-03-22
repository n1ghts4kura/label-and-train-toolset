"""
batch_extract_frames.py — 批量 MP4 视频帧提取工具
──────────────────────────────────────────────────
批量将多个 MP4 视频按指定帧率提取为 JPG 图片

使用方式:
    python scripts/batch_extract_frames.py ./videos/
    python scripts/batch_extract_frames.py ./videos/ --fps 30 --interval 5
    python scripts/batch_extract_frames.py ./videos/ --output ./frames/
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def batch_extract(video_dir, output_dir=None, fps=None, interval=None,
                 start_frame=None, end_frame=None, quality=95, video_exts=None):
    """
    批量提取视频帧

    Args:
        video_dir: 视频目录
        output_dir: 输出目录（默认: 视频目录_frames/）
        fps: 指定帧率
        interval: 帧间隔
        start_frame: 起始帧
        end_frame: 结束帧
        quality: JPG 质量
        video_exts: 视频文件扩展名列表
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        print(f"[ERROR] 目录不存在: {video_dir}")
        return

    if video_exts is None:
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    # 查找所有视频文件
    video_files = []
    for ext in video_exts:
        video_files.extend(video_dir.glob(f"*{ext}"))
        video_files.extend(video_dir.glob(f"*{ext.upper()}"))

    if not video_files:
        print(f"[ERROR] 在 {video_dir} 中未找到视频文件")
        print(f"支持的格式: {', '.join(video_exts)}")
        return

    print("=" * 60)
    print(f"  批量视频帧提取工具")
    print("=" * 60)
    print(f"  视频目录: {video_dir}")
    print(f"  找到 {len(video_files)} 个视频文件:")
    for vf in video_files:
        print(f"    - {vf.name}")
    print("=" * 60)

    # 获取脚本自身路径
    script_path = Path(__file__).parent / "extract_frames.py"

    # 批量处理
    success_count = 0
    fail_count = 0

    for video_path in sorted(video_files):
        print(f"\n>>> 处理: {video_path.name}")

        # 构建输出目录
        if output_dir:
            vid_output = Path(output_dir) / video_path.stem
        else:
            vid_output = video_path.stem + "_frames"

        # 构建命令
        cmd = [
            sys.executable,
            str(script_path),
            str(video_path),
            "-o", str(vid_output),
            "--quality", str(quality)
        ]

        if fps is not None:
            cmd.extend(["--fps", str(fps)])
        if interval is not None:
            cmd.extend(["--interval", str(interval)])
        if start_frame is not None:
            cmd.extend(["--start", str(start_frame)])
        if end_frame is not None:
            cmd.extend(["--end", str(end_frame)])

        # 执行
        try:
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode == 0:
                success_count += 1
                print(f"    [OK] {vid_output}")
            else:
                fail_count += 1
                print(f"    [FAIL] {video_path.name}")
        except Exception as e:
            fail_count += 1
            print(f"    [ERROR] {e}")

    print("\n" + "=" * 60)
    print(f"  完成！成功: {success_count}, 失败: {fail_count}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="批量 MP4 视频帧提取为 JPG 图片"
    )
    parser.add_argument("video_dir", type=str, help="视频文件或目录路径")
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="输出目录（默认: 每个视频单独一个文件夹）"
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

    batch_extract(
        video_dir=args.video_dir,
        output_dir=args.output,
        fps=args.fps,
        interval=args.interval,
        start_frame=args.start,
        end_frame=args.end,
        quality=args.quality
    )


if __name__ == "__main__":
    main()

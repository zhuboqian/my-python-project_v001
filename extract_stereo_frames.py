#!/usr/bin/env python3
"""Extract synchronized continuous frames from a recorded stereo MP4 pair."""

import argparse
import csv
import math
import re
from pathlib import Path

import cv2


DEFAULT_VIDEO_ROOT = Path("/home/nvidia/Documents/bincular_vision/dynamic_videos")


def parse_time(value):
    """Parse SS, MM:SS or HH:MM:SS with optional milliseconds into seconds."""
    text = str(value).strip()
    if not text:
        raise ValueError("Time value cannot be empty")
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid time: {value}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid time: {value}") from exc

    if len(numbers) == 1:
        seconds = numbers[0]
    elif len(numbers) == 2:
        minutes, second = numbers
        if not 0 <= second < 60:
            raise ValueError(f"Seconds must be in [0, 60): {value}")
        seconds = minutes * 60 + second
    else:
        hours, minutes, second = numbers
        if not 0 <= minutes < 60 or not 0 <= second < 60:
            raise ValueError(f"Minutes/seconds must be in [0, 60): {value}")
        seconds = hours * 3600 + minutes * 60 + second
    if seconds < 0:
        raise ValueError("Time cannot be negative")
    return seconds


def format_time(seconds):
    total_ms = round(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}-{minutes:02d}-{secs:02d}-{millis:03d}"


def latest_session(root):
    sessions = sorted(
        (path for path in root.glob("stereo_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not sessions:
        raise FileNotFoundError(f"No stereo recording directory found under {root}")
    return sessions[0]


def find_video_pair(session):
    left_files = sorted(session.glob("left_*.mp4"))
    right_files = sorted(session.glob("right_*.mp4"))
    if len(left_files) != 1 or len(right_files) != 1:
        raise FileNotFoundError(
            f"Expected one left and one right MP4 in {session}; "
            f"found left={len(left_files)}, right={len(right_files)}"
        )
    return left_files[0], right_files[0]


def video_info(capture, path):
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return fps, frame_count, width, height


def safe_name(path):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(path))


def extract_pair(session, start_seconds, end_seconds, output=None):
    left_video, right_video = find_video_pair(session)
    left_cap = cv2.VideoCapture(str(left_video))
    right_cap = cv2.VideoCapture(str(right_video))

    try:
        left_info = video_info(left_cap, left_video)
        right_info = video_info(right_cap, right_video)
        left_fps, left_count, left_w, left_h = left_info
        right_fps, right_count, right_w, right_h = right_info

        if abs(left_fps - right_fps) > 1e-3:
            raise RuntimeError(f"Left/right FPS mismatch: {left_fps} vs {right_fps}")
        if left_count != right_count:
            raise RuntimeError(f"Left/right frame-count mismatch: {left_count} vs {right_count}")
        if (left_w, left_h) != (right_w, right_h):
            raise RuntimeError(
                f"Left/right resolution mismatch: {left_w}x{left_h} vs {right_w}x{right_h}"
            )

        fps = left_fps
        duration = left_count / fps
        if end_seconds <= start_seconds:
            raise ValueError("End time must be greater than start time")
        if start_seconds >= duration:
            raise ValueError(
                f"Start time {start_seconds:.3f}s exceeds video duration {duration:.3f}s"
            )
        if end_seconds > duration:
            print(
                f"Warning: end time {end_seconds:.3f}s exceeds duration "
                f"{duration:.3f}s; clipping to video end"
            )
            end_seconds = duration

        # Zero-based and inclusive. Ceil prevents selecting a frame before start;
        # floor selects the final frame whose nominal timestamp is within the range.
        start_frame = max(0, math.ceil(start_seconds * fps - 1e-9))
        end_frame = min(left_count - 1, math.floor(end_seconds * fps + 1e-9))
        if end_frame < start_frame:
            raise ValueError("The selected interval contains no video frames")

        if output is None:
            clip_name = f"clip_{format_time(start_seconds)}_to_{format_time(end_seconds)}"
            output = session / "extracted_frames" / clip_name
        output = Path(output)
        if output.exists():
            raise FileExistsError(
                f"Output directory already exists: {output}\n"
                "Choose another --output path or remove the old extraction explicitly."
            )
        left_dir = output / "left"
        right_dir = output / "right"
        left_dir.mkdir(parents=True)
        right_dir.mkdir(parents=True)

        left_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        right_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        rows = []

        for frame_index in range(start_frame, end_frame + 1):
            left_ok, left_frame = left_cap.read()
            right_ok, right_frame = right_cap.read()
            if not left_ok or not right_ok:
                raise RuntimeError(
                    f"Decode failed at source frame {frame_index}: "
                    f"left={left_ok}, right={right_ok}"
                )

            pair_index = frame_index - start_frame + 1
            left_name = f"left_{pair_index:06d}_source_{frame_index:06d}.png"
            right_name = f"right_{pair_index:06d}_source_{frame_index:06d}.png"
            left_path = left_dir / left_name
            right_path = right_dir / right_name
            if not cv2.imwrite(str(left_path), left_frame):
                raise RuntimeError(f"Failed to save {left_path}")
            if not cv2.imwrite(str(right_path), right_frame):
                raise RuntimeError(f"Failed to save {right_path}")

            rows.append({
                "pair_index": pair_index,
                "source_frame_index_0based": frame_index,
                "source_frame_number_1based": frame_index + 1,
                "video_time_ms": round(frame_index * 1000.0 / fps, 3),
                "left_image": str(Path("left") / left_name),
                "right_image": str(Path("right") / right_name),
            })

        csv_path = output / "frames.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print("\nExtraction completed")
        print(f"  Session       : {session}")
        print(f"  Left video    : {left_video.name}")
        print(f"  Right video   : {right_video.name}")
        print(f"  Video FPS     : {fps:.3f}")
        print(f"  Video duration: {duration:.3f} s")
        print(f"  Frame range   : {start_frame} - {end_frame} (zero-based, inclusive)")
        print(f"  Stereo pairs  : {len(rows)}")
        print(f"  Output        : {output}")
        return output
    finally:
        left_cap.release()
        right_cap.release()


def extract_by_frame_numbers(session, start_frame, end_frame, output=None):
    """Extract an inclusive range using the 1-based Frame number in the video."""
    if start_frame < 1 or end_frame < 1:
        raise ValueError("Frame numbers start at 1")
    if end_frame < start_frame:
        raise ValueError("End Frame must be greater than or equal to Start Frame")

    left_video, _ = find_video_pair(session)
    capture = cv2.VideoCapture(str(left_video))
    try:
        fps, frame_count, _, _ = video_info(capture, left_video)
    finally:
        capture.release()

    if start_frame > frame_count:
        raise ValueError(
            f"Start Frame {start_frame} exceeds video frame count {frame_count}"
        )
    if end_frame > frame_count:
        raise ValueError(
            f"End Frame {end_frame} exceeds video frame count {frame_count}"
        )

    if output is None:
        output = (
            session
            / "extracted_frames"
            / f"clip_frame_{start_frame:06d}_to_{end_frame:06d}"
        )

    # Recorder overlay is 1-based: Frame 000001 is encoded frame index 0.
    start_seconds = (start_frame - 1) / fps
    end_seconds = (end_frame - 1) / fps
    return extract_pair(session, start_seconds, end_seconds, output)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract synchronized continuous PNG frames from ZED stereo videos"
    )
    parser.add_argument(
        "--session",
        type=Path,
        help="Recording session directory; defaults to the latest stereo_* directory",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
        help=f"Recording root used to find the latest session (default: {DEFAULT_VIDEO_ROOT})",
    )
    parser.add_argument(
        "--start-frame", type=int, help="First displayed Frame number (1-based, inclusive)"
    )
    parser.add_argument(
        "--end-frame", type=int, help="Last displayed Frame number (1-based, inclusive)"
    )
    parser.add_argument("--output", type=Path, help="Optional output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    session = args.session if args.session else latest_session(args.root)
    if not session.is_dir():
        raise NotADirectoryError(f"Session directory does not exist: {session}")

    start_frame = args.start_frame
    end_frame = args.end_frame
    if start_frame is None:
        start_frame = int(input("Start Frame (for example 1): ").strip())
    if end_frame is None:
        end_frame = int(input("End Frame   (for example 20): ").strip())
    extract_by_frame_numbers(session, start_frame, end_frame, args.output)


if __name__ == "__main__":
    main()

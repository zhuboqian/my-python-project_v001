#!/usr/bin/env python3
"""Extract selected synchronized left/right frames from a stereo recording."""

import argparse
import csv
from pathlib import Path

import cv2


DEFAULT_VIDEO_ROOT = Path("/home/nvidia/Documents/bincular_vision/dynamic_videos")


def latest_session(root: Path) -> Path:
    sessions = sorted(
        (path for path in root.glob("stereo_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not sessions:
        raise FileNotFoundError(f"No stereo recording directory found under {root}")
    return sessions[0]


def find_video_pair(session: Path):
    left_files = sorted(session.glob("left_*.mp4"))
    right_files = sorted(session.glob("right_*.mp4"))
    if len(left_files) != 1 or len(right_files) != 1:
        raise FileNotFoundError(
            f"Expected one left_*.mp4 and one right_*.mp4 in {session}; "
            f"found left={len(left_files)}, right={len(right_files)}"
        )
    return left_files[0], right_files[0]


def parse_frame_spec(value: str):
    """Parse values such as '1,5,20-25,100'. Frame numbers are 1-based."""
    frames = set()
    for token in value.replace("，", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            try:
                start, end = (int(part.strip()) for part in parts)
            except ValueError as exc:
                raise ValueError(f"Invalid frame range: {token}") from exc
            if start < 1 or end < start:
                raise ValueError(f"Invalid frame range: {token}")
            frames.update(range(start, end + 1))
        else:
            try:
                frame = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid frame number: {token}") from exc
            if frame < 1:
                raise ValueError("Frame numbers must start at 1")
            frames.add(frame)
    if not frames:
        raise ValueError("No frame numbers were provided")
    return sorted(frames)


def video_metadata(capture, path: Path):
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or count <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return fps, count, width, height


def read_frame(capture, frame_number: int, side: str):
    frame_index = frame_number - 1
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(f"Failed to decode {side} Frame {frame_number:06d}")
    return image


def extract_selected(session: Path, frame_numbers, output: Path):
    left_video, right_video = find_video_pair(session)
    left_cap = cv2.VideoCapture(str(left_video))
    right_cap = cv2.VideoCapture(str(right_video))

    try:
        left_info = video_metadata(left_cap, left_video)
        right_info = video_metadata(right_cap, right_video)
        left_fps, left_count, left_w, left_h = left_info
        right_fps, right_count, right_w, right_h = right_info

        if abs(left_fps - right_fps) > 1e-3:
            raise RuntimeError(f"Left/right FPS mismatch: {left_fps} vs {right_fps}")
        if left_count != right_count:
            raise RuntimeError(
                f"Left/right frame-count mismatch: {left_count} vs {right_count}"
            )
        if (left_w, left_h) != (right_w, right_h):
            raise RuntimeError(
                f"Left/right resolution mismatch: {left_w}x{left_h} vs "
                f"{right_w}x{right_h}"
            )

        invalid = [number for number in frame_numbers if number > left_count]
        if invalid:
            raise ValueError(
                f"Frame number exceeds video length ({left_count}): {invalid}"
            )

        output = output.resolve()
        left_dir = output / "left"
        right_dir = output / "right"
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for frame_number in frame_numbers:
            left_image = read_frame(left_cap, frame_number, "left")
            right_image = read_frame(right_cap, frame_number, "right")
            filename = f"frame_{frame_number:06d}.png"
            left_path = left_dir / filename
            right_path = right_dir / filename
            if not cv2.imwrite(str(left_path), left_image):
                raise RuntimeError(f"Failed to save: {left_path}")
            if not cv2.imwrite(str(right_path), right_image):
                raise RuntimeError(f"Failed to save: {right_path}")

            rows.append(
                {
                    "displayed_frame_number": frame_number,
                    "encoded_frame_index_0based": frame_number - 1,
                    "video_time_ms": round((frame_number - 1) * 1000.0 / left_fps, 3),
                    "left_image": str(Path("left") / filename),
                    "right_image": str(Path("right") / filename),
                }
            )

        csv_path = output / "selected_frames.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print("\nExtraction completed")
        print(f"  Session       : {session}")
        print(f"  Left video    : {left_video.name}")
        print(f"  Right video   : {right_video.name}")
        print(f"  Selected      : {len(frame_numbers)} stereo pairs")
        print(f"  Frame numbers : {', '.join(map(str, frame_numbers))}")
        print(f"  Output        : {output}")
        print(f"  Index CSV     : {csv_path}")
    finally:
        left_cap.release()
        right_cap.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract selected synchronized frames from left/right stereo videos"
    )
    parser.add_argument(
        "--session",
        type=Path,
        help="Folder containing one left_*.mp4 and one right_*.mp4; latest by default",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
        help=f"Recording root (default: {DEFAULT_VIDEO_ROOT})",
    )
    parser.add_argument(
        "--frames",
        help='Displayed Frame numbers, e.g. "1,5,20-25,100"',
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output folder; defaults to <session>/selected_frames",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    interactive_frame_spec = None
    if args.session:
        session = args.session
    else:
        default_session = latest_session(args.root)
        session_text = input(
            "Stereo video folder\n"
            f"[Press Enter for latest: {default_session}]\n> "
        ).strip()
        if session_text:
            try:
                # A frequent interactive input is to enter frame numbers at the
                # first prompt. Accept that form and use the latest recording.
                parse_frame_spec(session_text)
            except ValueError:
                session = Path(session_text).expanduser()
            else:
                interactive_frame_spec = session_text
                session = default_session
                print(f"Using latest stereo video folder: {session}")
        else:
            session = default_session
    if not session.is_dir():
        raise NotADirectoryError(f"Session directory does not exist: {session}")

    frame_spec = args.frames or interactive_frame_spec
    if not frame_spec:
        frame_spec = input('Frames (for example "1,5,20-25,100"): ').strip()
    frame_numbers = parse_frame_spec(frame_spec)

    if args.output:
        output = args.output
    else:
        default_output = session / "selected_frames"
        output_text = input(
            "Output folder\n"
            f"[Press Enter for: {default_output}]\n> "
        ).strip()
        output = Path(output_text).expanduser() if output_text else default_output
    extract_selected(session, frame_numbers, output)


if __name__ == "__main__":
    main()

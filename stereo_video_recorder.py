#!/usr/bin/env python3
"""Record synchronized, timestamped left/right video from a ZED X camera.

The camera setup follows takephotos.py: HD1200, 15 FPS and DEPTH_MODE.NONE.
Press Q in the preview window to stop recording safely.
"""

import argparse
import importlib
import site
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

def import_zed_sdk():
    """Import pyzed, including the user site hidden by some virtualenvs."""
    try:
        return importlib.import_module("pyzed.sl"), None
    except ImportError as first_error:
        user_site = site.getusersitepackages()
        if user_site and user_site not in sys.path:
            sys.path.append(user_site)
        try:
            return importlib.import_module("pyzed.sl"), None
        except ImportError:
            return None, first_error


sl, ZED_IMPORT_ERROR = import_zed_sdk()


DEFAULT_OUTPUT = Path("/home/nvidia/Documents/bincular_vision/dynamic_videos")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ZED X synchronized left/right timestamped video recorder"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output root directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--fps", type=int, default=15, help="Camera/video FPS")
    parser.add_argument(
        "--no-preview", action="store_true", help="Record without a preview window"
    )
    return parser.parse_args()


def to_bgr(image):
    """Convert a ZED image (normally BGRA) into BGR for OpenCV VideoWriter."""
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def draw_timestamp(frame, captured_at, frame_id):
    """Draw wall-clock time with milliseconds and a frame number at bottom-right."""
    result = frame.copy()
    height, width = result.shape[:2]
    timestamp = captured_at.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    lines = (timestamp, f"Frame: {frame_id:06d}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.65, min(width, height) / 1500.0)
    thickness = max(1, round(scale * 2))
    padding = 12
    line_gap = 10
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    box_width = max(size[0] for size in sizes) + 2 * padding
    box_height = sum(size[1] for size in sizes) + line_gap + 2 * padding
    x1 = max(0, width - box_width - 16)
    y1 = max(0, height - box_height - 16)

    overlay = result.copy()
    cv2.rectangle(overlay, (x1, y1), (width - 16, height - 16), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, result, 0.38, 0, result)

    text_top = y1 + padding
    for line, (text_width, text_height) in zip(lines, sizes):
        y = text_top + text_height
        x = width - 16 - padding - text_width
        cv2.putText(
            result,
            line,
            (x, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        text_top = y + line_gap
    return result


def draw_recording_status(frame, state):
    """Draw the recorder state and keyboard controls in the preview."""
    colors = {
        "READY": (0, 215, 255),
        "RECORDING": (0, 0, 255),
        "PAUSED": (0, 165, 255),
    }
    color = colors[state]
    result = frame.copy()
    cv2.rectangle(result, (16, 16), (620, 100), (0, 0, 0), -1)
    cv2.putText(
        result, state, (32, 52), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        result, "S: Start   P: Pause/Resume   Q: Finish",
        (32, 84), cv2.FONT_HERSHEY_SIMPLEX,
        0.58, (255, 255, 255), 1, cv2.LINE_AA,
    )
    if state == "RECORDING":
        cv2.circle(result, (594, 43), 10, (0, 0, 255), -1)
    return result


def create_writer(path, fps, frame_size):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), frame_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video file: {path}")
    return writer


def open_camera(fps):
    if sl is None:
        raise RuntimeError(
            "pyzed.sl is unavailable. Run this script in the same ZED SDK Python "
            f"environment used by takephotos.py. Original error: {ZED_IMPORT_ERROR}"
        )

    cameras = sl.Camera.get_device_list()
    if not cameras:
        raise RuntimeError("No ZED camera detected")

    print(f"Detected {len(cameras)} ZED camera(s)")
    for index, camera in enumerate(cameras):
        print(f"  Camera {index}: serial number {camera.serial_number}")

    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_serial_number(cameras[0].serial_number)
    init.camera_resolution = sl.RESOLUTION.HD1200
    init.camera_fps = fps
    init.depth_mode = sl.DEPTH_MODE.NONE

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise RuntimeError(f"Failed to open ZED camera: {status}")
    return zed


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")

    zed = None
    left_writer = None
    right_writer = None
    session_dir = None
    left_path = None
    right_path = None
    frame_id = 0
    failed_grabs = 0
    active_started = None
    active_elapsed = 0.0
    recording_started = False
    state = "READY"

    try:
        zed = open_camera(args.fps)
        left_mat = sl.Mat()
        right_mat = sl.Mat()

        print("\nCamera preview started")
        if not args.no_preview:
            print("  S: start recording")
            print("  P: pause/resume recording")
            print("  Q: finish and exit")
        else:
            print("  No-preview mode starts recording immediately; press Ctrl+C to stop")

        while True:
            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                failed_grabs += 1
                continue

            # One grab followed by two retrievals keeps the stereo pair synchronized.
            captured_at = datetime.now()
            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
            left = to_bgr(left_mat.get_data())
            right = to_bgr(right_mat.get_data())
            if left is None or right is None:
                failed_grabs += 1
                continue
            if left.shape[:2] != right.shape[:2]:
                raise RuntimeError(
                    f"Left/right image size mismatch: {left.shape[:2]} vs {right.shape[:2]}"
                )

            # In headless mode there is no keyboard, so start on the first frame.
            if args.no_preview and state == "READY":
                state = "START_REQUESTED"

            if state == "START_REQUESTED":
                session_name = datetime.now().strftime("stereo_%Y%m%d_%H%M%S")
                session_dir = args.output / session_name
                session_dir.mkdir(parents=True, exist_ok=False)
                left_path = session_dir / f"left_{session_name}.mp4"
                right_path = session_dir / f"right_{session_name}.mp4"
                height, width = left.shape[:2]
                frame_size = (width, height)
                left_writer = create_writer(left_path, args.fps, frame_size)
                right_writer = create_writer(right_path, args.fps, frame_size)
                recording_started = True
                state = "RECORDING"
                active_started = time.monotonic()
                print("\nRecording started")
                print(f"  Left video : {left_path}")
                print(f"  Right video: {right_path}")
                print(f"  Resolution : {width} x {height}")
                print(f"  Target FPS : {args.fps}")

            display_frame_id = frame_id + 1 if state == "RECORDING" else frame_id
            left_marked = draw_timestamp(left, captured_at, display_frame_id)
            right_marked = draw_timestamp(right, captured_at, display_frame_id)

            if state == "RECORDING":
                frame_id += 1
                left_writer.write(left_marked)
                right_writer.write(right_marked)

            if not args.no_preview:
                preview_width = 800
                preview_height = round(left_marked.shape[0] * preview_width / left_marked.shape[1])
                left_preview = cv2.resize(left_marked, (preview_width, preview_height))
                right_preview = cv2.resize(right_marked, (preview_width, preview_height))
                preview = cv2.hconcat((left_preview, right_preview))
                preview = draw_recording_status(preview, state)
                cv2.imshow("ZED X Stereo Recorder", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("s"), ord("S")) and state == "READY":
                    state = "START_REQUESTED"
                elif key in (ord("p"), ord("P")) and state == "RECORDING":
                    active_elapsed += time.monotonic() - active_started
                    active_started = None
                    state = "PAUSED"
                    print("Recording paused")
                elif key in (ord("p"), ord("P")) and state == "PAUSED":
                    state = "RECORDING"
                    active_started = time.monotonic()
                    print("Recording resumed")

    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    finally:
        if active_started is not None:
            active_elapsed += time.monotonic() - active_started
        if left_writer is not None:
            left_writer.release()
        if right_writer is not None:
            right_writer.release()
        if zed is not None:
            zed.close()
        cv2.destroyAllWindows()

        actual_fps = frame_id / active_elapsed if active_elapsed > 0 else 0.0
        if recording_started:
            print("\nRecording finished")
            print(f"  Saved stereo frames: {frame_id}")
            print(f"  Failed grabs       : {failed_grabs}")
            print(f"  Active duration    : {active_elapsed:.2f} s")
            print(f"  Average FPS        : {actual_fps:.2f}")
            print(f"  Output directory   : {session_dir}")


if __name__ == "__main__":
    main()
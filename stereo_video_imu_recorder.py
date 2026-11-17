#!/usr/bin/env python3
"""Record synchronized ZED X stereo video and IMU data for vibration tests.

Controls
--------
B: begin recording left/right video and IMU data
Q: stop recording (if active), save metadata, and exit

The camera frame timestamps and IMU timestamps both come from the ZED SDK clock.
This allows samples to be aligned afterwards without relying on GUI key timing.
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from stereo_video_recorder import (
    ZED_IMPORT_ERROR,
    create_writer,
    draw_timestamp,
    sl,
    to_bgr,
)


DEFAULT_OUTPUT = Path("/home/nvidia/Documents/bincular_vision/vibration_experiments")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record synchronized ZED X stereo video and six-axis IMU data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output root directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--fps", type=int, default=15, help="Stereo video FPS")
    parser.add_argument(
        "--preview-width",
        type=int,
        default=720,
        help="Displayed width of each eye image",
    )
    return parser.parse_args()


def timestamp_ns(timestamp):
    """Return a ZED Timestamp as integer nanoseconds."""
    return int(timestamp.get_nanoseconds())


def vector3(values):
    return float(values[0]), float(values[1]), float(values[2])


def draw_status(frame, recording, frame_count, imu_count):
    result = frame.copy()
    state = "RECORDING" if recording else "READY"
    color = (0, 0, 255) if recording else (0, 215, 255)
    cv2.rectangle(result, (16, 16), (680, 112), (0, 0, 0), -1)
    cv2.putText(
        result, state, (32, 50), cv2.FONT_HERSHEY_SIMPLEX,
        0.82, color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        result, "B: Begin recording    Q: Stop and save",
        (32, 79), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        result, f"Stereo frames: {frame_count}    IMU samples: {imu_count}",
        (32, 101), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (220, 220, 220), 1, cv2.LINE_AA,
    )
    if recording:
        cv2.circle(result, (650, 42), 10, (0, 0, 255), -1)
    return result


def open_camera(fps):
    if sl is None:
        raise RuntimeError(
            "pyzed.sl is unavailable. Run with the fistenv Python environment. "
            f"Original error: {ZED_IMPORT_ERROR}"
        )

    cameras = sl.Camera.get_device_list()
    if not cameras:
        raise RuntimeError("No ZED camera detected")

    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_serial_number(cameras[0].serial_number)
    init.camera_resolution = sl.RESOLUTION.HD1200
    init.camera_fps = fps
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.sensors_required = True

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise RuntimeError(f"Failed to open ZED camera: {status}")

    sensors = sl.SensorsData()
    sensor_status = zed.get_sensors_data(sensors, sl.TIME_REFERENCE.CURRENT)
    if sensor_status != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise RuntimeError(f"Camera opened, but IMU data is unavailable: {sensor_status}")

    return zed, int(cameras[0].serial_number)


def create_session(output_root, left_shape, fps):
    session_name = datetime.now().strftime("vibration_%Y%m%d_%H%M%S")
    session_dir = output_root / session_name
    session_dir.mkdir(parents=True, exist_ok=False)

    height, width = left_shape[:2]
    left_path = session_dir / f"left_{session_name}.mp4"
    right_path = session_dir / f"right_{session_name}.mp4"
    frames_path = session_dir / "frames.csv"
    imu_path = session_dir / "imu.csv"

    left_writer = create_writer(left_path, fps, (width, height))
    right_writer = create_writer(right_path, fps, (width, height))
    frames_file = frames_path.open("w", newline="", encoding="utf-8")
    imu_file = imu_path.open("w", newline="", encoding="utf-8")

    frame_csv = csv.writer(frames_file)
    frame_csv.writerow([
        "frame_id", "camera_timestamp_ns", "host_timestamp_ns",
        "elapsed_s", "wall_clock_time",
    ])
    imu_csv = csv.writer(imu_file)
    imu_csv.writerow([
        "sample_id", "imu_timestamp_ns", "host_timestamp_ns", "elapsed_s",
        "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
        "gyro_x_deg_s", "gyro_y_deg_s", "gyro_z_deg_s", "effective_rate_hz",
    ])

    return {
        "name": session_name,
        "dir": session_dir,
        "left_path": left_path,
        "right_path": right_path,
        "left_writer": left_writer,
        "right_writer": right_writer,
        "frames_file": frames_file,
        "imu_file": imu_file,
        "frame_csv": frame_csv,
        "imu_csv": imu_csv,
        "width": width,
        "height": height,
    }


def close_session(session):
    if not session:
        return
    session["left_writer"].release()
    session["right_writer"].release()
    session["frames_file"].flush()
    session["imu_file"].flush()
    session["frames_file"].close()
    session["imu_file"].close()


def main():
    args = parse_args()
    if args.fps <= 0 or args.preview_width <= 0:
        raise ValueError("--fps and --preview-width must be greater than zero")

    zed = None
    session = None
    recording = False
    frame_count = 0
    imu_count = 0
    failed_grabs = 0
    duplicate_imu_samples = 0
    last_imu_timestamp = -1
    record_start_camera_ns = None
    record_start_host_ns = None
    record_start_wall = None
    record_start_monotonic = None
    record_stop_wall = None

    try:
        zed, serial_number = open_camera(args.fps)
        camera_info = zed.get_camera_information()
        left_mat = sl.Mat()
        right_mat = sl.Mat()

        print(f"ZED X ready (serial: {serial_number})")
        print("  B: begin synchronized video + IMU recording")
        print("  Q: stop, save, and exit")

        while True:
            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                failed_grabs += 1
                continue

            host_ns = time.time_ns()
            captured_at = datetime.now()
            camera_ns = timestamp_ns(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE))
            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
            left = to_bgr(left_mat.get_data())
            right = to_bgr(right_mat.get_data())
            if left is None or right is None:
                failed_grabs += 1
                continue

            # Drain every batch even before recording, preventing stale samples from
            # being included when B is pressed.
            sensor_batch = []
            batch_status = zed.get_sensors_data_batch(sensor_batch)

            if recording:
                frame_count += 1
                elapsed_s = (camera_ns - record_start_camera_ns) / 1e9
                left_saved = draw_timestamp(left, captured_at, frame_count)
                right_saved = draw_timestamp(right, captured_at, frame_count)
                session["left_writer"].write(left_saved)
                session["right_writer"].write(right_saved)
                session["frame_csv"].writerow([
                    frame_count, camera_ns, host_ns, f"{elapsed_s:.9f}",
                    captured_at.isoformat(timespec="milliseconds"),
                ])

                if batch_status == sl.ERROR_CODE.SUCCESS:
                    for sensor_data in sensor_batch:
                        imu = sensor_data.get_imu_data()
                        if not imu.is_available:
                            continue
                        imu_ns = timestamp_ns(imu.timestamp)
                        if imu_ns < record_start_camera_ns:
                            continue
                        if imu_ns <= last_imu_timestamp:
                            duplicate_imu_samples += 1
                            continue
                        last_imu_timestamp = imu_ns
                        acc_x, acc_y, acc_z = vector3(imu.get_linear_acceleration())
                        gyr_x, gyr_y, gyr_z = vector3(imu.get_angular_velocity())
                        imu_count += 1
                        session["imu_csv"].writerow([
                            imu_count, imu_ns, time.time_ns(),
                            f"{(imu_ns - record_start_camera_ns) / 1e9:.9f}",
                            f"{acc_x:.9f}", f"{acc_y:.9f}", f"{acc_z:.9f}",
                            f"{gyr_x:.9f}", f"{gyr_y:.9f}", f"{gyr_z:.9f}",
                            f"{float(imu.effective_rate):.3f}",
                        ])

                if frame_count % 30 == 0:
                    session["frames_file"].flush()
                    session["imu_file"].flush()
                    if imu_count == 0:
                        print("WARNING: no IMU samples have been recorded")

            shown_id = frame_count if recording else 0
            left_preview = draw_timestamp(left, captured_at, shown_id)
            right_preview = draw_timestamp(right, captured_at, shown_id)
            preview_height = round(left.shape[0] * args.preview_width / left.shape[1])
            left_preview = cv2.resize(left_preview, (args.preview_width, preview_height))
            right_preview = cv2.resize(right_preview, (args.preview_width, preview_height))
            preview = cv2.hconcat((left_preview, right_preview))
            preview = draw_status(preview, recording, frame_count, imu_count)
            cv2.imshow("ZED X Vibration Video + IMU Recorder", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                record_stop_wall = datetime.now()
                break
            if key in (ord("b"), ord("B")) and not recording:
                session = create_session(args.output, left.shape, args.fps)
                recording = True
                record_start_camera_ns = camera_ns
                record_start_host_ns = time.time_ns()
                record_start_wall = datetime.now()
                record_start_monotonic = time.monotonic()
                print("\nRecording started")
                print(f"  Output: {session['dir']}")

    except KeyboardInterrupt:
        record_stop_wall = datetime.now()
        print("\nInterrupted; saving recorded data")
    finally:
        duration_s = (
            time.monotonic() - record_start_monotonic
            if record_start_monotonic is not None else 0.0
        )
        close_session(session)
        if zed is not None:
            zed.close()
        cv2.destroyAllWindows()

        if session is not None:
            metadata = {
                "experiment_type": "vibration_table_stereo_video_and_imu",
                "camera_model": str(camera_info.camera_model),
                "camera_serial_number": serial_number,
                "resolution": [session["width"], session["height"]],
                "configured_video_fps": args.fps,
                "saved_stereo_frames": frame_count,
                "saved_imu_samples": imu_count,
                "failed_image_grabs": failed_grabs,
                "discarded_duplicate_imu_samples": duplicate_imu_samples,
                "start_wall_clock": record_start_wall.isoformat(timespec="milliseconds"),
                "stop_wall_clock": (
                    record_stop_wall or datetime.now()
                ).isoformat(timespec="milliseconds"),
                "start_camera_timestamp_ns": record_start_camera_ns,
                "start_host_timestamp_ns": record_start_host_ns,
                "duration_s": duration_s,
                "measured_video_fps": frame_count / duration_s if duration_s else 0.0,
                "measured_imu_rate_hz": imu_count / duration_s if duration_s else 0.0,
                "coordinate_convention": "ZED IMAGE coordinate system (x right, y down, z forward)",
                "acceleration_unit": "m/s^2 (includes gravity)",
                "angular_velocity_unit": "deg/s",
                "synchronization": (
                    "frames.csv and imu.csv use ZED device timestamps in nanoseconds; "
                    "align each frame with the nearest IMU timestamp"
                ),
                "files": {
                    "left_video": session["left_path"].name,
                    "right_video": session["right_path"].name,
                    "frame_timestamps": "frames.csv",
                    "imu_samples": "imu.csv",
                },
            }
            metadata_path = session["dir"] / "session_metadata.json"
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("\nRecording saved")
            print(f"  Directory     : {session['dir']}")
            print(f"  Stereo frames : {frame_count}")
            print(f"  IMU samples   : {imu_count}")
            print(f"  Video FPS     : {metadata['measured_video_fps']:.2f}")
            print(f"  IMU rate      : {metadata['measured_imu_rate_hz']:.2f} Hz")
        else:
            print("No recording was started; no experiment files were created")


if __name__ == "__main__":
    main()

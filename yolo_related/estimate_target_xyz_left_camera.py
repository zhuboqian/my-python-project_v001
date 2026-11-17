#!/usr/bin/env python3
"""Estimate target XYZ in the rectified left-camera coordinate system.

Pipeline: stereo rectification -> Model D detection on rectified left image ->
ROI-NCC disparity -> Z = fx*B/d -> pinhole back-projection for X and Y.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from compare_dynamic_stereo_matching import (  # noqa: E402
    detect_boxes,
    load_calibration,
    rectify_pair,
    roi_ncc_disparity,
)
from register_spdconv import load_yolo_with_spdconv  # noqa: E402

DEFAULT_MODEL = (
    HERE / "runs/detect/runs/train/"
    "final_model_D_3domain_independent_v2/weights/best.pt"
)
DEFAULT_CALIB = PROJECT / "stereo_calibration.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True, help="Left image")
    parser.add_argument("--right", type=Path, required=True, help="Right image")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--output", type=Path, default=HERE / "runs/detect/target_xyz")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-disparity", type=int, default=256)
    parser.add_argument("--y-tolerance", type=int, default=4)
    parser.add_argument("--min-ncc-score", type=float, default=0.55)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument(
        "--target-point", nargs=2, type=float, metavar=("U", "V"),
        help="When multiple boxes exist, select the one nearest this rectified-left pixel",
    )
    return parser.parse_args()


def choose_box(boxes: list[dict], target_point: list[float] | None) -> dict | None:
    if not boxes:
        return None
    if target_point is None:
        return boxes[0]  # detect_boxes already sorts by confidence
    u0, v0 = target_point
    return min(
        boxes,
        key=lambda box: np.hypot(
            (box["xyxy"][0] + box["xyxy"][2]) / 2.0 - u0,
            (box["xyxy"][1] + box["xyxy"][3]) / 2.0 - v0,
        ),
    )


def save_visualization(image: np.ndarray, result: dict, path: Path) -> None:
    canvas = image.copy()
    box = result.get("left_box_xyxy")
    if box:
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (44, 190, 90), 3)
        u, v = result["target_center_rectified_px"]
        cv2.drawMarker(canvas, (int(round(u)), int(round(v))), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)
        xyz = result.get("xyz_left_m")
        if xyz:
            text = f"X={xyz['X']:.3f}m Y={xyz['Y']:.3f}m Z={xyz['Z']:.3f}m"
        else:
            text = f"FAILED: {result['status']}"
        origin = (max(8, x1), max(30, y1 - 12))
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Failed to save visualization: {path}")


def main() -> None:
    args = parse_args()
    for path in (args.left, args.right, args.model, args.calib):
        if not path.is_file():
            raise FileNotFoundError(path)

    left = cv2.imread(str(args.left), cv2.IMREAD_COLOR)
    right = cv2.imread(str(args.right), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise RuntimeError("Failed to read one or both input images")

    calib = load_calibration(args.calib)
    raw = calib["raw"]
    projection = np.asarray(raw["P_left"], dtype=float)
    fx, fy = float(projection[0, 0]), float(projection[1, 1])
    cx, cy = float(projection[0, 2]), float(projection[1, 2])
    baseline = float(calib["baseline"])
    left_rect, right_rect = rectify_pair(left, right, calib)

    model = load_yolo_with_spdconv(str(args.model))
    boxes = detect_boxes(model, left_rect, args.conf, args.imgsz, args.device, args.iou)
    selected = choose_box(boxes, args.target_point)
    result = {
        "status": "no_left_detection",
        "coordinate_frame": "rectified_left_camera",
        "axis_convention": {"X": "right", "Y": "down", "Z": "forward"},
        "left_image": str(args.left.resolve()),
        "right_image": str(args.right.resolve()),
        "model": str(args.model.resolve()),
        "calibration": str(args.calib.resolve()),
        "calibration_parameters": {
            "fx_px": fx, "fy_px": fy, "cx_px": cx, "cy_px": cy,
            "baseline_m": baseline,
        },
        "settings": {
            "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
            "max_disparity_px": args.max_disparity,
            "y_tolerance_px": args.y_tolerance,
            "min_ncc_score": args.min_ncc_score,
        },
        "candidate_count": len(boxes),
    }

    if selected is not None:
        x1, y1, x2, y2 = selected["xyxy"]
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        result.update({
            "left_box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "left_detection_confidence": float(selected["conf"]),
            "target_center_rectified_px": [float(u), float(v)],
        })
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        disparity, score, status = roi_ncc_disparity(
            left_gray, right_gray, selected["xyxy"], args.max_disparity,
            args.y_tolerance, args.min_ncc_score, None, 16,
        )
        result.update({"status": status, "disparity_px": disparity, "ncc_score": score})
        if status == "ok" and disparity is not None and disparity > 0:
            z = fx * baseline / disparity
            if args.min_depth <= z <= args.max_depth:
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                result["status"] = "ok"
                result["xyz_left_m"] = {"X": float(x), "Y": float(y), "Z": float(z)}
                result["range_m"] = float(np.sqrt(x * x + y * y + z * z))
            else:
                result["status"] = "depth_out_of_range"

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.left.stem
    json_path = args.output / f"{stem}_xyz.json"
    image_path = args.output / f"{stem}_xyz.png"
    result["visualization"] = str(image_path.resolve())
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_visualization(left_rect, result, image_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"Image: {image_path}")
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

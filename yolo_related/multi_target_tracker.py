#!/usr/bin/env python3
"""Multi-target YOLO detection with lightweight temporal association.

The detector returns every target box. Tracks use a constant-velocity Kalman
filter and Hungarian assignment. A short detector miss is retained as a
*prediction*, never mislabeled as a real detection.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from register_spdconv import load_yolo_with_spdconv, register  # noqa: E402

register()

DEFAULT_MODEL = HERE / (
    "runs/detect/runs/train/final_model_D_3domain_independent_v2/weights/best.pt"
)
DEFAULT_OUTPUT = HERE / "runs/detect/multi_target_tracking"


def xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(float)
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], np.float32)


def cxcywh_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, width, height = state[:4].astype(float)
    width, height = max(2.0, width), max(2.0, height)
    return np.array([cx - width / 2, cy - height / 2,
                     cx + width / 2, cy + height / 2], np.float32)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


class KalmanBox:
    def __init__(self, box: np.ndarray):
        self.filter = cv2.KalmanFilter(8, 4)
        self.filter.transitionMatrix = np.eye(8, dtype=np.float32)
        self.filter.transitionMatrix[0, 4] = 1.0
        self.filter.transitionMatrix[1, 5] = 1.0
        self.filter.transitionMatrix[2, 6] = 1.0
        self.filter.transitionMatrix[3, 7] = 1.0
        self.filter.measurementMatrix = np.zeros((4, 8), np.float32)
        self.filter.measurementMatrix[:4, :4] = np.eye(4, dtype=np.float32)
        self.filter.processNoiseCov = np.diag(
            [1, 1, 1, 1, 10, 10, 4, 4]
        ).astype(np.float32) * 1e-2
        self.filter.measurementNoiseCov = np.eye(4, dtype=np.float32) * 2e-1
        self.filter.errorCovPost = np.eye(8, dtype=np.float32)
        self.filter.statePost[:4, 0] = xyxy_to_cxcywh(box)

    def predict(self) -> np.ndarray:
        return cxcywh_to_xyxy(self.filter.predict().ravel())

    def correct(self, box: np.ndarray) -> np.ndarray:
        measurement = xyxy_to_cxcywh(box).reshape(4, 1)
        return cxcywh_to_xyxy(self.filter.correct(measurement).ravel())


@dataclass
class Track:
    track_id: int
    kalman: KalmanBox
    box: np.ndarray
    confidence: float
    hits: int = 1
    missed: int = 0
    detected_this_frame: bool = True


class MultiTargetTracker:
    def __init__(self, max_age=3, min_hits=2, max_center_distance=0.12,
                 minimum_iou=0.01):
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_center_distance = max_center_distance
        self.minimum_iou = minimum_iou
        self.next_id = 1
        self.tracks: list[Track] = []

    @staticmethod
    def _assignment(cost: np.ndarray):
        if cost.size == 0:
            return np.empty((0, 2), dtype=int)
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(cost)
            return np.column_stack((rows, cols))
        except Exception:
            # Dependency-free fallback: globally greedy assignment.
            available_rows, available_cols, pairs = set(range(cost.shape[0])), set(range(cost.shape[1])), []
            while available_rows and available_cols:
                choice = min(((cost[r, c], r, c) for r in available_rows for c in available_cols),
                             key=lambda item: item[0])
                _, row, col = choice
                pairs.append((row, col)); available_rows.remove(row); available_cols.remove(col)
            return np.asarray(pairs, dtype=int)

    def update(self, detections: np.ndarray, scores: np.ndarray,
               frame_shape: tuple[int, int]) -> list[Track]:
        height, width = frame_shape
        diagonal = float(np.hypot(width, height))
        predicted = []
        for track in self.tracks:
            track.box = track.kalman.predict()
            track.detected_this_frame = False
            predicted.append(track.box.copy())

        cost = np.full((len(self.tracks), len(detections)), 1e6, dtype=np.float32)
        for row, track_box in enumerate(predicted):
            tc = xyxy_to_cxcywh(track_box)[:2]
            for col, detection in enumerate(detections):
                dc = xyxy_to_cxcywh(detection)[:2]
                distance = float(np.linalg.norm(tc - dc) / max(diagonal, 1.0))
                overlap = box_iou(track_box, detection)
                if distance <= self.max_center_distance or overlap >= self.minimum_iou:
                    cost[row, col] = 0.55 * distance / max(self.max_center_distance, 1e-6) + 0.45 * (1 - overlap)

        matched_tracks, matched_detections = set(), set()
        for row, col in self._assignment(cost):
            if cost[row, col] >= 1e5:
                continue
            track = self.tracks[int(row)]
            track.box = track.kalman.correct(detections[int(col)])
            track.confidence = float(scores[int(col)])
            track.hits += 1; track.missed = 0; track.detected_this_frame = True
            matched_tracks.add(int(row)); matched_detections.add(int(col))

        for row, track in enumerate(self.tracks):
            if row not in matched_tracks:
                track.missed += 1
        self.tracks = [track for track in self.tracks if track.missed <= self.max_age]

        for col, detection in enumerate(detections):
            if col in matched_detections:
                continue
            self.tracks.append(Track(
                self.next_id, KalmanBox(detection), detection.copy(), float(scores[col])
            ))
            self.next_id += 1
        return self.tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video path or image directory")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.02)
    parser.add_argument("--iou", type=float, default=0.30)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-age", type=int, default=3,
                        help="Keep a prediction for this many consecutive missed frames")
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--fps", type=float, default=15.0,
                        help="Output FPS for an image-directory source")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def source_frames(source: str, directory_fps: float):
    path = Path(source)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
        if not files:
            raise RuntimeError(f"No images in {path}")
        for index, file in enumerate(files):
            image = cv2.imread(str(file))
            if image is None:
                continue
            yield index, file.name, image, directory_fps
        return
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or directory_fps
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        yield index, f"frame_{index:06d}", image, fps
        index += 1
    capture.release()


def draw_tracks(image: np.ndarray, tracks: list[Track], min_hits: int) -> np.ndarray:
    canvas = image.copy()
    for track in tracks:
        if track.hits < min_hits and not track.detected_this_frame:
            continue
        x1, y1, x2, y2 = np.round(track.box).astype(int)
        detected = track.detected_this_frame
        color = (40, 190, 80) if detected else (0, 165, 255)
        state = "DET" if detected else f"PRED-{track.missed}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track.track_id} {state} {track.confidence:.2f}"
        cv2.putText(canvas, label, (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_yolo_with_spdconv(str(args.model))
    tracker = MultiTargetTracker(max_age=args.max_age, min_hits=args.min_hits)
    records, writer = [], None
    output_video = args.output / "multi_target_tracking.mp4"

    for frame_index, source_name, frame, fps in source_frames(args.source, args.fps):
        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                               device=args.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            boxes = np.empty((0, 4), dtype=np.float32); scores = np.empty(0, dtype=np.float32)
        else:
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        tracks = tracker.update(boxes, scores, frame.shape[:2])
        annotated = draw_tracks(frame, tracks, args.min_hits)
        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                     float(fps), (width, height))
        writer.write(annotated)
        for track in tracks:
            if track.hits < args.min_hits and not track.detected_this_frame:
                continue
            x1, y1, x2, y2 = map(float, track.box)
            records.append({
                "frame_index": frame_index, "source_name": source_name,
                "track_id": track.track_id,
                "source": "detection" if track.detected_this_frame else "prediction",
                "missed_frames": track.missed, "hits": track.hits,
                "confidence": track.confidence, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
        if args.show:
            cv2.imshow("Multi-target tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if (frame_index + 1) % 100 == 0:
            print(f"Processed {frame_index + 1} frames", flush=True)

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if records:
        with (args.output / "tracks.csv").open("w", newline="", encoding="utf-8-sig") as f:
            csv_writer = csv.DictWriter(f, fieldnames=list(records[0]))
            csv_writer.writeheader(); csv_writer.writerows(records)
    report = {
        "source": args.source, "model": str(args.model),
        "settings": vars(args) | {"model": str(args.model), "output": str(args.output)},
        "output_video": str(output_video), "track_rows": len(records),
        "note": "DET is a YOLO observation; PRED-n is Kalman-only and must not be counted as a detection.",
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

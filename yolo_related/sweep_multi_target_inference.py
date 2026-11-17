#!/usr/bin/env python3
"""Sweep inference settings for final Model D on the frozen multi-target set."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from register_spdconv import load_yolo_with_spdconv, register  # noqa: E402

register()

MODEL = HERE / "runs/detect/runs/train/final_model_D_3domain_independent_v2/weights/best.pt"
DATASET = HERE / "runs/detect/multi_target_model_D_20261117/dataset"
OUT = HERE / "runs/detect/multi_target_model_D_sweep_20261117"
IMAGE_DIR = DATASET / "images/test"
LABEL_DIR = DATASET / "labels/test"
SIZES = (640, 960, 1280)
NMS_IOUS = (0.30, 0.50, 0.70)
CONFIDENCES = (0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25)


def load_truth(image: Path) -> np.ndarray:
    h, w = cv2.imread(str(image)).shape[:2]
    boxes = []
    for line in (LABEL_DIR / f"{image.stem}.txt").read_text().splitlines():
        if not line.strip():
            continue
        _, xc, yc, bw, bh = map(float, line.split())
        boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h,
                      (xc + bw / 2) * w, (yc + bh / 2) * h])
    return np.asarray(boxes, dtype=np.float32)


def iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def match(pred_boxes: np.ndarray, scores: np.ndarray, truth: np.ndarray, threshold=0.5):
    order = np.argsort(-scores)
    unmatched = list(range(len(truth)))
    tp = 0
    for index in order:
        if not unmatched:
            break
        values = iou_one(pred_boxes[index], truth[unmatched])
        best = int(np.argmax(values))
        if values[best] >= threshold:
            tp += 1
            unmatched.pop(best)
    return tp, len(pred_boxes) - tp, len(truth) - tp


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    images = sorted(IMAGE_DIR.glob("*.png"))
    truths = {p.stem: load_truth(p) for p in images}
    model = load_yolo_with_spdconv(str(MODEL))
    rows = []

    for size in SIZES:
        for nms_iou in NMS_IOUS:
            print(f"[RUN] imgsz={size}, NMS IoU={nms_iou:.2f}", flush=True)
            outputs = model.predict(
                source=[str(p) for p in images], imgsz=size, conf=min(CONFIDENCES),
                iou=nms_iou, device=0, verbose=False, stream=False,
            )
            cached = {}
            for image, result in zip(images, outputs):
                if result.boxes is None or len(result.boxes) == 0:
                    boxes = np.empty((0, 4), dtype=np.float32)
                    scores = np.empty(0, dtype=np.float32)
                else:
                    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
                cached[image.stem] = (boxes, scores)

            for conf in CONFIDENCES:
                tp = fp = fn = 0
                full = zero = predicted = 0
                for image in images:
                    boxes, scores = cached[image.stem]
                    keep = scores >= conf
                    boxes_c, scores_c = boxes[keep], scores[keep]
                    tpi, fpi, fni = match(boxes_c, scores_c, truths[image.stem])
                    tp += tpi; fp += fpi; fn += fni; predicted += len(boxes_c)
                    full += int(fni == 0)
                    zero += int(len(boxes_c) == 0)
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                rows.append({
                    "imgsz": size, "nms_iou": nms_iou, "conf": conf,
                    "tp": tp, "fp": fp, "fn": fn, "predicted_boxes": predicted,
                    "precision_iou50": precision, "recall_iou50": recall,
                    "f1_iou50": f1, "all_targets_detected_images": full,
                    "all_targets_detected_rate": full / len(images),
                    "zero_prediction_images": zero,
                })

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "sweep_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    best_f1 = max(rows, key=lambda r: (r["f1_iou50"], r["recall_iou50"]))
    best_recall_p80 = max(
        (r for r in rows if r["precision_iou50"] >= 0.80),
        key=lambda r: (r["recall_iou50"], r["f1_iou50"]), default=None,
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "model": str(MODEL), "images": len(images),
        "ground_truth_objects": int(sum(len(v) for v in truths.values())),
        "matching_iou": 0.50, "best_f1": best_f1,
        "best_recall_with_precision_ge_0.80": best_recall_p80,
        "tested": {"imgsz": SIZES, "nms_iou": NMS_IOUS, "conf": CONFIDENCES},
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

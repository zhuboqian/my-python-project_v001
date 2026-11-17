#!/usr/bin/env python3
"""Evaluate final Model D on the frozen real multi-target stereo capture."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from register_spdconv import load_yolo_with_spdconv, register  # noqa: E402

register()

SOURCE = Path("/home/nvidia/Documents/bincular_vision/multi_target_data/raw")
MODEL = HERE / "runs/detect/runs/train/final_model_D_3domain_independent_v2/weights/best.pt"
OUT = HERE / "runs/detect/multi_target_model_D_20261117"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)

    image_dir = OUT / "dataset/images/test"
    label_dir = OUT / "dataset/labels/test"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    rows, counts = [], []
    right_by_id = {
        p.name.replace("right_", "", 1): p for p in (SOURCE / "right").glob("*.png")
    }
    for image in sorted((SOURCE / "left").glob("*.png")):
        label = image.with_suffix(".txt")
        pair_id = image.name.replace("left_", "", 1)
        right = right_by_id.get(pair_id)
        if not label.is_file() or right is None:
            raise RuntimeError(f"Incomplete pair or label: {image}")
        objects = sum(bool(line.strip()) for line in label.read_text().splitlines())
        counts.append(objects)
        for source, destination in ((image, image_dir / image.name),
                                    (label, label_dir / label.name)):
            if not destination.exists():
                os.symlink(source, destination)
        rows.append({
            "pair_id": pair_id.removesuffix(".png"), "left_image": str(image),
            "right_image": str(right), "left_label": str(label),
            "object_count": objects,
        })

    with (OUT / "image_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    data_yaml = OUT / "dataset/data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(OUT / "dataset"), "train": "images/test", "val": "images/test",
        "test": "images/test", "nc": 1, "names": {0: "target"},
    }, sort_keys=False), encoding="utf-8")

    model = load_yolo_with_spdconv(str(MODEL))
    metrics = model.val(
        data=str(data_yaml), split="test", imgsz=640, batch=8, conf=0.001,
        iou=0.65, single_cls=True, device=0, workers=0, plots=True,
        project=str(OUT), name="validation", exist_ok=True,
    )
    # Save an inspectable set of detections at the deployment threshold.
    model.predict(
        source=str(image_dir), imgsz=640, conf=0.05, iou=0.50, device=0,
        save=True, save_txt=True, save_conf=True, verbose=False,
        project=str(OUT), name="predictions_conf005", exist_ok=True,
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "frozen real multi-target left-image detection",
        "stereo_pair_count": len(rows), "left_image_count": len(rows),
        "annotation_count": sum(counts),
        "targets_per_image": dict(sorted(Counter(counts).items())),
        "model": str(MODEL), "model_sha256": digest(MODEL),
        "evaluation": {"imgsz": 640, "batch": 8, "conf": 0.001, "iou": 0.65},
        "deployment_preview": {"conf": 0.05, "iou": 0.50},
        "metrics": {
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
        },
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

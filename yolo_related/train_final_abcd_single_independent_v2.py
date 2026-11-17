#!/usr/bin/env python3
"""Controlled final A-D single-Original-domain training on independent_v2."""

from __future__ import annotations

import gc
import csv
import io
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from register_spdconv import load_yolo_with_spdconv, register  # noqa: E402


register()
SEED = 42
EPOCHS = 100
DATA = HERE / "datasets/original_independent_v2/data.yaml"
PROJECT = HERE / "runs/detect/runs/train"
INIT_ROOT = PROJECT / "final_abcd_independent_v2_initializations"
STATE = PROJECT / "final_abcd_single_independent_v2_state.json"
MODELS = {
    "A": "YOLOv8n baseline",
    "B": "P2",
    "C": "SPDConv+P2+C2fAttn",
    "D": "P2+illumination residual gate",
}


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def completed_epochs(run: Path) -> int:
    result = run / "results.csv"
    if not result.is_file():
        return 0
    text = result.read_bytes().replace(b"\x00", b"").decode("utf-8-sig", errors="replace")
    epochs = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            epochs.append(int(float(row['epoch'])))
        except (ValueError, TypeError, KeyError):
            continue
    return max(epochs, default=0)


def common_args(name: str) -> dict:
    return dict(
        data=str(DATA), epochs=EPOCHS, imgsz=640, batch=8, device="cuda", workers=0,
        optimizer="AdamW", lr0=0.001, lrf=0.01, momentum=0.937,
        weight_decay=0.0005, warmup_epochs=3, warmup_momentum=0.8,
        warmup_bias_lr=0.1, patience=100, save=True, save_period=10, val=True,
        project=str(PROJECT), name=name, exist_ok=True, seed=SEED,
        deterministic=True, single_cls=True, close_mosaic=10, amp=True,
        mixup=0.0, copy_paste=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5, erasing=0.4,
        plots=True, verbose=True,
    )


def write_state(active=None, status="running") -> None:
    payload = {
        "updated_at": datetime.now().isoformat(),
        "status": status,
        "active_model": active,
        "data": str(DATA),
        "epochs": EPOCHS,
        "models": {},
    }
    for label, description in MODELS.items():
        run = PROJECT / f"final_model_{label}_single_independent_v2"
        payload["models"][label] = {
            "description": description,
            "completed_epochs": completed_epochs(run),
            "run": str(run),
            "best": str(run / "weights/best.pt"),
        }
    STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train_one(label: str, description: str) -> None:
    name = f"final_model_{label}_single_independent_v2"
    run = PROJECT / name
    done = completed_epochs(run)
    if done >= EPOCHS:
        print(f"[{label}] complete, skip: {done}/{EPOCHS}", flush=True)
        return
    write_state(label)
    if done:
        last = run / "weights/last.pt"
        if not last.is_file():
            raise FileNotFoundError(last)
        print(f"[{label}] resume {done}/{EPOCHS}: {last}", flush=True)
        model = load_yolo_with_spdconv(str(last))
        model.train(resume=True, workers=0)
    else:
        init = INIT_ROOT / f"model_{label}_seed42_pretrained_init.pt"
        if not init.is_file():
            raise FileNotFoundError(init)
        seed_all()
        print(f"[{label}] start {description} from audited initialization: {init}", flush=True)
        model = load_yolo_with_spdconv(str(init))
        model.train(**common_args(name))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    write_state(None)


def main() -> None:
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run with conda fistenv on the Jetson host")
    print(f"CUDA: {torch.cuda.get_device_name(0)}; data={DATA}", flush=True)
    for label, description in MODELS.items():
        train_one(label, description)
    write_state(None, "complete")
    print("Final A/B/C/D independent_v2 single-domain training complete.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Controlled final A/B/C/D training on leakage-free three-domain v2 data."""
from __future__ import annotations

import gc
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from register_spdconv import create_yolo_from_yaml, load_yolo_with_spdconv, register

register()
SEED = 42
EPOCHS = 100
DATA = Path("/home/nvidia/Documents/bincular_vision/training/combined_independent_v2/data.yaml")
PRETRAINED = HERE / "yolov8n.pt"
PROJECT = HERE / "runs/detect/runs/train"
INIT_ROOT = PROJECT / "final_abcd_independent_v2_initializations"
STATE = PROJECT / "final_abcd_independent_v2_state.json"
MODELS = {
    # New paper labels: A=old A, B=old C, C=old F, D=old H.
    "A": (HERE / "yolov8_orignial.yaml", "YOLOv8n baseline"),
    "B": (HERE / "yolov8_p2_only.yaml", "P2"),
    "C": (HERE / "yolov8_designed.yaml", "SPDConv+P2+C2fAttn"),
    "D": (HERE / "yolov8_p2_illumination_gate_h.yaml", "P2+illumination residual gate"),
}


def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def completed_epochs(run: Path) -> int:
    path = run / "results.csv"
    if not path.is_file(): return 0
    text = path.read_bytes().replace(b"\x00", b"").decode("utf-8-sig", errors="replace")
    return max(0, sum(bool(line.strip()) for line in text.splitlines()) - 1)


def build_initialization(label: str, yaml: Path) -> Path:
    output = INIT_ROOT / f"model_{label}_seed42_pretrained_init.pt"
    if output.is_file(): return output
    INIT_ROOT.mkdir(parents=True, exist_ok=True)
    seed_all()
    model = create_yolo_from_yaml(str(yaml))
    model.load(str(PRETRAINED))
    model.save(str(output))
    del model; gc.collect(); torch.cuda.empty_cache()
    return output


def common_args(name: str):
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


def write_state(active=None, status="running"):
    payload = {
        "updated_at": datetime.now().isoformat(), "status": status,
        "active_model": active, "data": str(DATA), "epochs": EPOCHS,
        "models": {},
    }
    for label, (_, description) in MODELS.items():
        run = PROJECT / f"final_model_{label}_3domain_independent_v2"
        payload["models"][label] = {
            "description": description, "completed_epochs": completed_epochs(run),
            "run": str(run), "best": str(run / "weights/best.pt"),
        }
    STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train_one(label, yaml, description):
    name = f"final_model_{label}_3domain_independent_v2"
    run = PROJECT / name
    done = completed_epochs(run)
    if done >= EPOCHS:
        print(f"[{label}] complete, skip: {done}/{EPOCHS}", flush=True); return
    write_state(label)
    if done:
        last = run / "weights/last.pt"
        if not last.is_file(): raise FileNotFoundError(last)
        print(f"[{label}] resume {done}/{EPOCHS}: {last}", flush=True)
        model = load_yolo_with_spdconv(str(last))
        model.train(resume=True, workers=0)
    else:
        init = build_initialization(label, yaml)
        print(f"[{label}] start {description} from {init}", flush=True)
        model = load_yolo_with_spdconv(str(init))
        model.train(**common_args(name))
    del model; gc.collect(); torch.cuda.empty_cache(); write_state(None)


def main():
    for path in (DATA, PRETRAINED, *(item[0] for item in MODELS.values())):
        if not path.is_file(): raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run with conda fistenv on the Jetson host")
    print(f"CUDA: {torch.cuda.get_device_name(0)}; data={DATA}", flush=True)
    for label, (yaml, description) in MODELS.items():
        train_one(label, yaml, description)
    write_state(None, "complete")
    print("Final A/B/C/D independent-v2 three-domain training complete.", flush=True)


if __name__ == "__main__":
    main()

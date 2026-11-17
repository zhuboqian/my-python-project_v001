#!/usr/bin/env python3
"""Audit and build a leakage-free three-domain split without modifying legacy data.

Default mode is audit-only. Pass --materialize to create symlinks in OUTPUT.
Grouping deliberately ignores domain and camera side, so all derivatives of one
physical acquisition stay in one split. Frozen-test source ids are excluded.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

LEGACY = Path("/home/nvidia/Documents/bincular_vision/training/combined")
TESTING = Path("/home/nvidia/Documents/bincular_vision/testing")
OUTPUT = Path("/home/nvidia/Documents/bincular_vision/training/combined_independent_v2")
AUDIT_OUTPUT = Path(__file__).resolve().parent / "dataset_audits/combined_independent_v2"
DOMAINS = ("original", "tension", "degraded_physics")
TEST_SCENES = ("original", "tension_heavy", "light_clutter_heavy")
SEED = 42
VAL_FRACTION = 0.15
ADJACENT_GAP_SECONDS = 10
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def normalized_id(stem: str) -> str:
    for prefix in ("tension_", "degraded_physics_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    for prefix in ("left_", "right_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem


def source_domain(path: Path) -> str:
    target = str(path.resolve(strict=False))
    for domain in DOMAINS:
        if f"/training/{domain}/" in target:
            return domain
    return "unknown"


def side_of(path: Path) -> str:
    stem = path.resolve(strict=False).stem
    return "left" if stem.startswith("left_") else "right" if stem.startswith("right_") else "unknown"


def frozen_ids() -> set[str]:
    ids = set()
    for scene in TEST_SCENES:
        for side in ("left", "right"):
            folder = TESTING / scene / side
            if folder.is_dir():
                ids.update(normalized_id(path.stem) for path in folder.iterdir()
                           if path.suffix.lower() in IMAGE_SUFFIXES)
    return ids


def acquisition_blocks(source_ids: set[str]) -> dict[str, str]:
    """Group temporally adjacent captures; unknown names remain singleton groups."""
    parsed, mapping = [], {}
    for source_id in source_ids:
        match = re.fullmatch(r"(\d{8})_(\d{6})_(\d+)", source_id)
        if not match:
            mapping[source_id] = f"singleton:{source_id}"
            continue
        stamp = match.group(1) + match.group(2)
        from datetime import datetime
        parsed.append((datetime.strptime(stamp, "%Y%m%d%H%M%S"), source_id))
    parsed.sort()
    block_index, previous = 0, None
    for stamp, source_id in parsed:
        if (previous is None or stamp.date() != previous.date() or
                (stamp - previous).total_seconds() > ADJACENT_GAP_SECONDS):
            block_index += 1
        mapping[source_id] = f"capture_block_{block_index:04d}"
        previous = stamp
    return mapping


def collect_legacy():
    rows, rejected = [], []
    for old_split in ("train", "val"):
        image_dir = LEGACY / "images" / old_split
        label_dir = LEGACY / "labels" / old_split
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            row = {
                "legacy_split": old_split,
                "legacy_name": image.name,
                "image_link": str(image),
                "resolved_image": str(image.resolve(strict=False)),
                "source_id": normalized_id(image.stem),
                "domain": source_domain(image),
                "side": side_of(image),
            }
            label = label_dir / f"{image.stem}.txt"
            row["label_link"] = str(label)
            row["resolved_label"] = str(label.resolve(strict=False))
            reasons = []
            if not image.exists(): reasons.append("broken_image_link")
            if image.name.startswith("debug_"): reasons.append("debug_artifact")
            if row["domain"] == "unknown": reasons.append("unknown_domain")
            if row["side"] == "unknown": reasons.append("unknown_camera_side")
            if not label.exists(): reasons.append("missing_label")
            if reasons:
                rejected.append({**row, "reason": "+".join(reasons)})
            else:
                rows.append(row)
    return rows, rejected


def choose_validation_groups(groups: dict[str, list[dict]]) -> set[str]:
    rng = random.Random(SEED)
    keys = list(groups)
    total = Counter(row["domain"] for key in keys for row in groups[key])
    target = {domain: round(total[domain] * VAL_FRACTION) for domain in DOMAINS}
    target_group_count = max(1, round(len(keys) * VAL_FRACTION))
    best_selection, best_score = None, float("inf")
    # Fixed-cardinality seeded search avoids a validation set dominated by only
    # a few large acquisition sequences while balancing all three domains.
    for _ in range(20000):
        selected = set(rng.sample(keys, target_group_count))
        current = Counter(row["domain"] for key in selected for row in groups[key])
        score = sum(abs(target[d] - current[d]) for d in DOMAINS)
        if score < best_score:
            best_selection, best_score = selected, score
            if score == 0:
                break
    return best_selection or set()


def safe_link(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    AUDIT_OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, rejected = collect_legacy()
    frozen = frozen_ids()
    all_ids = {row["source_id"] for row in rows + rejected} | frozen
    block_of = acquisition_blocks(all_ids)
    frozen_blocks = {block_of[source_id] for source_id in frozen}
    kept, frozen_rejected = [], []
    for row in rows:
        row["capture_group"] = block_of[row["source_id"]]
        if row["capture_group"] in frozen_blocks:
            reason = ("frozen_test_source" if row["source_id"] in frozen
                      else "adjacent_to_frozen_test_source")
            frozen_rejected.append({**row, "reason": reason})
        else:
            kept.append(row)

    # If any representation of a physical acquisition is invalid, exclude the
    # complete acquisition to avoid asymmetric domain/eye composition.
    invalid_ids = {row["source_id"] for row in rejected}
    invalid_group_rejected = [
        {**row, "reason": "same_source_has_invalid_member"}
        for row in kept if row["source_id"] in invalid_ids
    ]
    kept = [row for row in kept if row["source_id"] not in invalid_ids]
    groups = defaultdict(list)
    for row in kept:
        groups[row["capture_group"]].append(row)
    val_ids = choose_validation_groups(groups)
    for row in kept:
        row["new_split"] = "val" if row["capture_group"] in val_ids else "train"

    train_ids = {r["capture_group"] for r in kept if r["new_split"] == "train"}
    actual_val_ids = {r["capture_group"] for r in kept if r["new_split"] == "val"}
    assert not (train_ids & actual_val_ids)
    assert not ((train_ids | actual_val_ids) & frozen_blocks)

    all_rejected = rejected + frozen_rejected + invalid_group_rejected
    summary = {
        "seed": SEED,
        "validation_fraction_target": VAL_FRACTION,
        "adjacent_capture_gap_seconds": ADJACENT_GAP_SECONDS,
        "legacy_candidate_entries": len(rows) + len(rejected),
        "accepted_images": len(kept),
        "accepted_capture_groups": len(groups),
        "accepted_source_ids": len({r["source_id"] for r in kept}),
        "train_images": sum(r["new_split"] == "train" for r in kept),
        "val_images": sum(r["new_split"] == "val" for r in kept),
        "train_capture_groups": len(train_ids),
        "val_capture_groups": len(actual_val_ids),
        "per_split_domain": {
            split: dict(Counter(r["domain"] for r in kept if r["new_split"] == split))
            for split in ("train", "val")
        },
        "rejected_by_reason": dict(Counter(r["reason"] for r in all_rejected)),
        "frozen_source_ids": len(frozen),
        "train_val_source_overlap": 0,
        "train_or_val_frozen_overlap": 0,
        "output": str(OUTPUT),
        "materialized": args.materialize,
    }

    fields = ["new_split", "capture_group", "source_id", "domain", "side", "legacy_split",
              "legacy_name", "resolved_image", "resolved_label"]
    with (AUDIT_OUTPUT / "accepted_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows({k: r[k] for k in fields} for r in kept)
    reject_fields = sorted({k for r in all_rejected for k in r})
    with (AUDIT_OUTPUT / "rejected_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=reject_fields)
        writer.writeheader(); writer.writerows(all_rejected)
    (AUDIT_OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.materialize:
        for row in kept:
            split, name = row["new_split"], row["legacy_name"]
            safe_link(Path(row["resolved_image"]), OUTPUT / "images" / split / name)
            safe_link(Path(row["resolved_label"]), OUTPUT / "labels" / split / f"{Path(name).stem}.txt")
        (OUTPUT / "data.yaml").write_text(
            f"path: {OUTPUT}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: target\n",
            encoding="utf-8")
        (OUTPUT / "split_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

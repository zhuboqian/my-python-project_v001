#!/usr/bin/env python3
"""Materialize and audit the final leakage-free dataset suite."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "dataset_audits/combined_independent_v2"
TRAINING = Path("/home/nvidia/Documents/bincular_vision/training")
TESTING = Path("/home/nvidia/Documents/bincular_vision/testing")
COMBINED = TRAINING / "combined_independent_v2"
SINGLE = TRAINING / "original_independent_v2"
SUITE = Path("/home/nvidia/Documents/bincular_vision/dataset_suite_independent_v2")
TEST_SCENES = ("original", "tension_heavy", "light_clutter_heavy")


def link(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source)


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm(stem: str):
    for prefix in ("tension_", "degraded_physics_"):
        if stem.startswith(prefix): stem = stem[len(prefix):]
    for prefix in ("left_", "right_"):
        if stem.startswith(prefix): stem = stem[len(prefix):]
    return stem


def load_manifest():
    with (AUDIT / "accepted_manifest.csv").open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_single(rows):
    chosen = [row for row in rows if row["domain"] == "original"]
    for row in chosen:
        split, name = row["new_split"], row["legacy_name"]
        link(Path(row["resolved_image"]), SINGLE / "images" / split / name)
        link(Path(row["resolved_label"]), SINGLE / "labels" / split / f"{Path(name).stem}.txt")
    (SINGLE / "data.yaml").write_text(
        f"path: {SINGLE}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: target\n",
        encoding="utf-8")
    return chosen


def freeze_tests():
    records = []
    for scene in TEST_SCENES:
        root = TESTING / scene
        image_stems = {p.stem for p in (root / "images").iterdir() if p.is_file()}
        for subset in ("left", "right"):
            files = sorted(p for p in (root / subset).glob("*") if p.is_file())
            for path in files:
                records.append({
                    "scene": scene, "subset": subset, "name": path.name,
                    "source_id": norm(path.stem), "size": path.stat().st_size,
                    "sha256": digest(path), "path": str(path),
                })
        labels = sorted(p for p in (root / "labels").glob("*.txt")
                        if p.stem in image_stems)
        for path in labels:
            records.append({
                "scene": scene, "subset": "labels", "name": path.name,
                "source_id": norm(path.stem), "size": path.stat().st_size,
                "sha256": digest(path), "path": str(path),
            })
        yaml_path = SUITE / "frozen_tests" / f"{scene}.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(
            f"path: {root}\ntest: images\nnc: 1\nnames:\n  0: target\n",
            encoding="utf-8")
    return records


def verify_links(root: Path):
    result = {}
    for split in ("train", "val"):
        images = list((root / "images" / split).iterdir())
        labels = list((root / "labels" / split).iterdir())
        result[split] = {
            "images": len(images), "labels": len(labels),
            "broken_images": sum(not p.exists() for p in images),
            "broken_labels": sum(not p.exists() for p in labels),
            "stem_mismatch": len({p.stem for p in images} ^ {p.stem for p in labels}),
        }
    return result


def main():
    SUITE.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()
    single = build_single(rows)
    frozen_records = freeze_tests()
    train_groups = {r["capture_group"] for r in rows if r["new_split"] == "train"}
    val_groups = {r["capture_group"] for r in rows if r["new_split"] == "val"}
    train_ids = {r["source_id"] for r in rows if r["new_split"] == "train"}
    val_ids = {r["source_id"] for r in rows if r["new_split"] == "val"}
    frozen_ids = {r["source_id"] for r in frozen_records}
    audit = {
        "policy": "grouped by cross-domain physical acquisition and <=10 s adjacent capture block",
        "combined": verify_links(COMBINED),
        "single_original": verify_links(SINGLE),
        "combined_domain_counts": {
            split: dict(Counter(r["domain"] for r in rows if r["new_split"] == split))
            for split in ("train", "val")
        },
        "single_counts": dict(Counter(r["new_split"] for r in single)),
        "frozen_test_counts": {
            scene: dict(Counter(r["subset"] for r in frozen_records if r["scene"] == scene))
            for scene in TEST_SCENES
        },
        "overlap_checks": {
            "train_vs_val_capture_groups": len(train_groups & val_groups),
            "train_vs_val_source_ids": len(train_ids & val_ids),
            "train_vs_frozen_source_ids": len(train_ids & frozen_ids),
            "val_vs_frozen_source_ids": len(val_ids & frozen_ids),
        },
        "training_yamls": {
            "single_original": str(SINGLE / "data.yaml"),
            "three_domain_full": str(COMBINED / "data.yaml"),
        },
        "frozen_test_yamls": {
            scene: str(SUITE / "frozen_tests" / f"{scene}.yaml") for scene in TEST_SCENES
        },
    }
    (SUITE / "dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    with (SUITE / "frozen_test_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(frozen_records[0]))
        writer.writeheader(); writer.writerows(frozen_records)
    (SUITE / "README.md").write_text(f"""# 独立数据集套件 v2

本套件用于修复旧数据中同源退化版本、左右目和相邻采集帧跨集合的问题。

## 正式训练配置

- 单域训练：`{SINGLE / 'data.yaml'}`
- 完整三域训练：`{COMBINED / 'data.yaml'}`

两套训练集沿用完全相同的采集块归属。单域数据是三域数据中 original 子集，不能单独重新随机划分。

## 冻结测试

三套冻结测试各包含46对双目图像。目标检测只使用 `images/` 中的46幅带标注左目图像；双目定位使用 `left/` 与 `right/` 的完整46对图像。

冻结测试不得用于训练、验证、阈值选择、早停或超参数调整。

## 历史数据

`training/original`、`training/combined`及其派生模型保留用于追溯，但不再作为修复数据泄漏后的论文最终训练数据。

完整数量、哈希和重叠检查见 `dataset_audit.json` 与 `frozen_test_manifest.csv`。
""", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

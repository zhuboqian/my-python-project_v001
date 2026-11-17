#!/usr/bin/env python3
"""Compare four stereo matching/depth pipelines on extracted dynamic frames.

P1: Model F left/right detections + bounding-box center disparity
P2: Model F left detection + ROI-SGBM
P3: Model F left detection + ROI-NCC template matching
P4: Model F left detection + multi-patch NCC + GMM filtering

All methods share the same rectified stereo frames and left Model F detection.
"""

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from register_spdconv import load_yolo_with_spdconv  # noqa: E402


DEFAULT_CLIP = Path(
    "/home/nvidia/Documents/bincular_vision/dynamic_videos/"
    "stereo_20260820_100225/extracted_frames/clip_frame_000173_to_000204"
)
DEFAULT_MODEL = SCRIPT_DIR / (
    "runs/detect/runs/train/target_detector_F_real_phone_v1_continue/weights/best.pt"
)
DEFAULT_CALIB = PROJECT_DIR / "stereo_calibration.pkl"
METHODS = ("center_disparity", "roi_sgbm", "roi_ncc", "multipatch_ncc_gmm")


def synchronize_if_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def timed_call(function, *args, **kwargs):
    synchronize_if_cuda()
    started = time.perf_counter()
    value = function(*args, **kwargs)
    synchronize_if_cuda()
    return value, (time.perf_counter() - started) * 1000.0


def load_calibration(path):
    # NumPy 2.x pickles reference numpy._core, while this project's .venv may
    # still use NumPy 1.x (numpy.core). Both contain the same required classes.
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
        sys.modules["numpy._core.numeric"] = np.core.numeric
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    map_h, map_w = raw["map_left_x"].shape
    if raw["map_right_x"].shape != (map_h, map_w):
        raise ValueError("Left/right calibration maps have different sizes")
    return {
        "raw": raw,
        "fx": float(raw["P_left"][0, 0]),
        "baseline": float(np.linalg.norm(raw["T"])),
        "img_w": map_w,
        "img_h": map_h,
        # Keys required by the existing GMM implementation.
        "map_left_x": raw["map_left_x"],
        "map_left_y": raw["map_left_y"],
        "map_right_x": raw["map_right_x"],
        "map_right_y": raw["map_right_y"],
    }


def load_pairs(clip):
    index_path = clip / "frames.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing frame index: {index_path}")
    with index_path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    pairs = []
    for row in rows:
        left_path = clip / row["left_image"]
        right_path = clip / row["right_image"]
        if not left_path.is_file() or not right_path.is_file():
            raise FileNotFoundError(f"Missing stereo pair: {left_path}, {right_path}")
        pairs.append((row, left_path, right_path))
    if not pairs:
        raise ValueError(f"No stereo pairs listed in {index_path}")
    return pairs


def rectify_pair(left, right, calib):
    expected = (calib["img_h"], calib["img_w"])
    if left.shape[:2] != expected or right.shape[:2] != expected:
        raise ValueError(
            f"Input must match calibration size {expected[::-1]}, got "
            f"left={left.shape[1::-1]}, right={right.shape[1::-1]}"
        )
    raw = calib["raw"]
    left_rect = cv2.remap(
        left, raw["map_left_x"], raw["map_left_y"], cv2.INTER_LINEAR
    )
    right_rect = cv2.remap(
        right, raw["map_right_x"], raw["map_right_y"], cv2.INTER_LINEAR
    )
    return left_rect, right_rect


def detect_boxes(model, image, conf, imgsz, device, iou=None):
    predict_kwargs = {
        "source": image, "conf": conf, "imgsz": imgsz,
        "device": device, "verbose": False,
    }
    if iou is not None:
        predict_kwargs["iou"] = iou
    result = model.predict(**predict_kwargs)[0]
    boxes = []
    if result.boxes is None:
        return boxes
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    for box, score in zip(xyxy, scores):
        boxes.append({"xyxy": box.astype(float).tolist(), "conf": float(score)})
    boxes.sort(key=lambda item: item["conf"], reverse=True)
    return boxes


def box_center(box):
    x1, y1, x2, y2 = box["xyxy"]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def select_left_box(boxes, previous_box=None, initial_point=None, max_jump=200.0):
    """Lock onto one target instead of switching to the highest-confidence box."""
    if not boxes:
        return None
    if previous_box is not None:
        previous_center = box_center(previous_box)
        ranked = sorted(
            boxes,
            key=lambda box: np.hypot(
                box_center(box)[0] - previous_center[0],
                box_center(box)[1] - previous_center[1],
            ),
        )
        distance = np.hypot(
            box_center(ranked[0])[0] - previous_center[0],
            box_center(ranked[0])[1] - previous_center[1],
        )
        return ranked[0] if distance <= max_jump else None
    if initial_point is not None:
        px, py = initial_point
        return min(
            boxes,
            key=lambda box: np.hypot(box_center(box)[0] - px, box_center(box)[1] - py),
        )
    return boxes[0]


def center_disparity(left_box, right_boxes, max_y_diff, max_disparity=None):
    if left_box is None or not right_boxes:
        return None, "missing_detection"
    lx1, ly1, lx2, ly2 = left_box["xyxy"]
    left_cx = (lx1 + lx2) / 2.0
    left_cy = (ly1 + ly2) / 2.0
    candidates = []
    for right_box in right_boxes:
        rx1, ry1, rx2, ry2 = right_box["xyxy"]
        right_cx = (rx1 + rx2) / 2.0
        right_cy = (ry1 + ry2) / 2.0
        disparity = left_cx - right_cx
        y_diff = abs(left_cy - right_cy)
        disparity_valid = disparity > 0 and (
            max_disparity is None or disparity <= max_disparity
        )
        if disparity_valid and y_diff <= max_y_diff:
            candidates.append((y_diff, -right_box["conf"], disparity))
    if not candidates:
        return None, "no_epipolar_pair"
    candidates.sort()
    return float(candidates[0][2]), "ok"


def clip_box(box, width, height, min_size=6):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(np.floor(x1))))
    y1 = max(0, min(height - 1, int(np.floor(y1))))
    x2 = max(0, min(width, int(np.ceil(x2))))
    y2 = max(0, min(height, int(np.ceil(y2))))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return x1, y1, x2, y2


def roi_ncc_disparity(left_gray, right_gray, bbox, max_disparity, y_tolerance,
                      min_score, coarse_disparity=None, search_radius=16):
    height, width = left_gray.shape
    clipped = clip_box(bbox, width, height)
    if clipped is None:
        return None, None, "small_roi"
    x1, y1, x2, y2 = clipped
    bw, bh = x2 - x1, y2 - y1
    margin = max(2, round(min(bw, bh) * 0.15))
    tx1, ty1 = max(0, x1 - margin), max(0, y1 - margin)
    tx2, ty2 = min(width, x2 + margin), min(height, y2 + margin)
    template = left_gray[ty1:ty2, tx1:tx2]
    template_h, template_w = template.shape

    if coarse_disparity is None:
        search_x1 = max(0, tx1 - max_disparity)
        search_x2 = min(width, tx2 + 2)
    else:
        expected_x1 = tx1 - coarse_disparity
        search_x1 = max(0, int(np.floor(expected_x1 - search_radius)))
        search_x2 = min(
            width, int(np.ceil(expected_x1 + search_radius + template_w + 1))
        )
    search_y1 = max(0, ty1 - y_tolerance)
    search_y2 = min(height, ty2 + y_tolerance)
    search = right_gray[search_y1:search_y2, search_x1:search_x2]
    if search.shape[0] < template_h or search.shape[1] < template_w:
        return None, None, "invalid_search_roi"
    if float(np.std(template)) < 2.0:
        return None, None, "low_texture"

    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(response)
    if not np.isfinite(score) or score < min_score:
        return None, float(score), "low_ncc_score"
    matched_x1 = search_x1 + location[0]
    left_center_x = (tx1 + tx2) / 2.0
    right_center_x = matched_x1 + template_w / 2.0
    disparity = left_center_x - right_center_x
    if disparity <= 0 or disparity > max_disparity:
        return None, float(score), "invalid_disparity"
    return float(disparity), float(score), "ok"


def roi_sgbm_disparity(left_gray, right_gray, bbox, max_disparity,
                       coarse_disparity=None, search_radius=16):
    height, width = left_gray.shape
    clipped = clip_box(bbox, width, height)
    if clipped is None:
        return None, "small_roi"
    x1, y1, x2, y2 = clipped
    bw, bh = x2 - x1, y2 - y1
    margin_x = max(24, bw)
    margin_y = max(12, bh)
    crop_x1 = max(0, x1 - max_disparity - margin_x)
    crop_x2 = min(width, x2 + margin_x)
    crop_y1 = max(0, y1 - margin_y)
    crop_y2 = min(height, y2 + margin_y)
    left_roi = left_gray[crop_y1:crop_y2, crop_x1:crop_x2]
    right_roi = right_gray[crop_y1:crop_y2, crop_x1:crop_x2]

    available_width = left_roi.shape[1]
    if coarse_disparity is None:
        min_disparity = 0
        num_disparities = min(
            max_disparity, max(16, ((available_width - 16) // 16) * 16)
        )
        num_disparities = (num_disparities // 16) * 16
    else:
        # Jetson OpenCV 4.13 can underflow its internal allocation when
        # StereoSGBM uses a non-zero minDisparity on a cropped ROI. Compute a
        # compact zero-based range and retain only coarse +/- radius values.
        min_disparity = 0
        required_upper = coarse_disparity + search_radius
        span = max(16, int(np.ceil(required_upper / 16.0)) * 16)
        num_disparities = min(
            span, max(16, ((available_width - 16) // 16) * 16)
        )
    if num_disparities < 16 or left_roi.shape[0] < 20:
        return None, "small_sgbm_roi"

    block_size = 5
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size ** 2,
        P2=32 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=50,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity_map = matcher.compute(left_roi, right_roi).astype(np.float32) / 16.0
    bx1, bx2 = x1 - crop_x1, x2 - crop_x1
    by1, by2 = y1 - crop_y1, y2 - crop_y1
    target_disp = disparity_map[by1:by2, bx1:bx2]
    upper_disparity = min_disparity + num_disparities
    valid = target_disp[
        (target_disp > max(1.0, min_disparity))
        & (target_disp <= upper_disparity)
    ]
    if coarse_disparity is not None:
        valid = valid[
            (valid >= coarse_disparity - search_radius)
            & (valid <= coarse_disparity + search_radius)
        ]
    if valid.size < 5:
        return None, "insufficient_valid_disparity"
    q1, q3 = np.percentile(valid, (25, 75))
    iqr = q3 - q1
    robust = valid[(valid >= q1 - 1.5 * iqr) & (valid <= q3 + 1.5 * iqr)]
    if robust.size < 3:
        robust = valid
    return float(np.median(robust)), "ok"


def full_frame_sgbm_disparity(left_gray, right_gray, bbox, max_disparity,
                              coarse_disparity=None, search_radius=16):
    """Compute dense full-frame SGBM, then read robust disparity in target box."""
    height, width = left_gray.shape
    clipped = clip_box(bbox, width, height)
    if clipped is None:
        return None, "small_roi"
    x1, y1, x2, y2 = clipped
    num_disparities = min(
        max_disparity, max(16, ((width - 16) // 16) * 16)
    )
    num_disparities = (num_disparities // 16) * 16
    if num_disparities < 16 or height < 20:
        return None, "small_full_frame"

    block_size = 5
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size ** 2,
        P2=32 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=50,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity_map = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    target_disp = disparity_map[y1:y2, x1:x2]
    valid = target_disp[(target_disp > 1.0) & (target_disp <= num_disparities)]
    if coarse_disparity is not None:
        valid = valid[
            (valid >= coarse_disparity - search_radius)
            & (valid <= coarse_disparity + search_radius)
        ]
    if valid.size < 5:
        return None, "insufficient_valid_disparity"
    q1, q3 = np.percentile(valid, (25, 75))
    iqr = q3 - q1
    robust = valid[(valid >= q1 - 1.5 * iqr) & (valid <= q3 + 1.5 * iqr)]
    if robust.size < 3:
        robust = valid
    return float(np.median(robust)), "ok"


def multipatch_ncc_gmm_disparity(left_gray, right_gray, bbox, max_disparity,
                                 min_score, coarse_disparity=None,
                                 search_radius=16):
    """Estimate disparity from 3x3 NCC patches and OpenCV two-component GMM."""
    height, width = left_gray.shape
    clipped = clip_box(bbox, width, height, min_size=9)
    if clipped is None:
        return None, None, "small_roi"
    x1, y1, x2, y2 = clipped
    bw, bh = x2 - x1, y2 - y1
    margin = max(3, min(bw, bh) // 3)
    rx1, ry1 = max(0, x1 - margin), max(0, y1 - margin)
    rx2, ry2 = min(width, x2 + margin), min(height, y2 + margin)
    cell_w, cell_h = (rx2 - rx1) // 3, (ry2 - ry1) // 3
    if cell_w < 3 or cell_h < 3:
        return None, None, "small_patch"

    disparities, scores = [], []
    for row in range(3):
        for col in range(3):
            px1 = rx1 + col * cell_w
            py1 = ry1 + row * cell_h
            px2 = rx2 if col == 2 else px1 + cell_w
            py2 = ry2 if row == 2 else py1 + cell_h
            patch = left_gray[py1:py2, px1:px2]
            ph, pw = patch.shape
            if ph < 3 or pw < 3 or float(np.std(patch)) < 2.0:
                continue
            if coarse_disparity is None:
                sx1 = max(0, px1 - max_disparity)
                sx2 = min(width, px2 + 2)
            else:
                expected_x1 = px1 - coarse_disparity
                sx1 = max(0, int(np.floor(expected_x1 - search_radius)))
                sx2 = min(
                    width, int(np.ceil(expected_x1 + search_radius + pw + 1))
                )
            search = right_gray[py1:py2, sx1:sx2]
            if search.shape[1] < pw:
                continue
            response = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(response)
            disparity = (px1 + px2) / 2.0 - (sx1 + location[0] + pw / 2.0)
            if score >= min_score and 0 < disparity <= max_disparity:
                disparities.append(float(disparity))
                scores.append(float(score))

    if len(disparities) < 3:
        return None, None, "insufficient_patch_matches"

    disp_array = np.asarray(disparities, dtype=np.float32)
    q1, q3 = np.percentile(disp_array, (25, 75))
    iqr = q3 - q1
    keep = (disp_array >= q1 - 1.5 * iqr) & (disp_array <= q3 + 1.5 * iqr)
    disp_array = disp_array[keep]
    score_array = np.asarray(scores, dtype=np.float32)[keep]
    if disp_array.size < 3:
        return None, None, "insufficient_gmm_samples"

    inlier_disparities = disp_array
    if disp_array.size >= 5:
        # Normalize both dimensions so disparity scale does not dominate score.
        disp_std = max(float(np.std(disp_array)), 1e-6)
        score_std = max(float(np.std(score_array)), 1e-6)
        features = np.column_stack((
            (disp_array - np.mean(disp_array)) / disp_std,
            (score_array - np.mean(score_array)) / score_std,
        )).astype(np.float32)
        try:
            em = cv2.ml.EM_create()
            em.setClustersNumber(2)
            em.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_GENERIC)
            em.setTermCriteria((cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-4))
            success, _, labels, _ = em.trainEM(features)
            if success:
                labels = labels.ravel().astype(int)
                candidates = []
                for label in (0, 1):
                    values = disp_array[labels == label]
                    cluster_scores = score_array[labels == label]
                    if values.size >= 2:
                        candidates.append((float(np.var(values)), -float(np.mean(cluster_scores)), values))
                if candidates:
                    candidates.sort(key=lambda item: (item[0], item[1]))
                    inlier_disparities = candidates[0][2]
        except cv2.error:
            inlier_disparities = disp_array

    disparity = float(np.median(inlier_disparities))
    return disparity, float(np.mean(score_array)), "ok"


def depth_from_disparity(disparity, calib, min_depth, max_depth):
    if disparity is None or disparity <= 0:
        return None, "invalid_disparity"
    depth = calib["fx"] * calib["baseline"] / disparity
    if not min_depth <= depth <= max_depth:
        return None, "depth_out_of_range"
    return float(depth), "ok"


def blank_method():
    return {
        "depth_m": None,
        "disparity_px": None,
        "match_score": None,
        "status": "not_run",
        "match_ms": None,
        "total_ms": None,
    }


def method_result(disparity, status, match_ms, shared_ms, calib, args, score=None):
    result = blank_method()
    result.update({
        "disparity_px": disparity,
        "match_score": score,
        "status": status,
        "match_ms": match_ms,
        "total_ms": shared_ms + match_ms,
    })
    if status == "ok":
        depth, depth_status = depth_from_disparity(
            disparity, calib, args.min_depth, args.max_depth
        )
        result["depth_m"] = depth
        result["status"] = depth_status
    return result


def process_frame(model, left, right, calib, args, previous_left_box=None):
    rectify_start = time.perf_counter()
    left_rect, right_rect = rectify_pair(left, right, calib)
    rectify_ms = (time.perf_counter() - rectify_start) * 1000.0

    left_boxes, left_detect_ms = timed_call(
        detect_boxes, model, left_rect, args.conf, args.imgsz, args.device
    )
    left_box = select_left_box(
        left_boxes,
        previous_left_box,
        tuple(args.initial_point) if args.initial_point else None,
        args.max_track_jump,
    )
    methods = {name: blank_method() for name in METHODS}
    if left_box is None:
        for result in methods.values():
            result["status"] = "no_left_detection"
        return methods, None, rectify_ms, left_detect_ms, 0.0

    left_xyxy = left_box["xyxy"]
    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    # P1 requires a second full-image Model F inference on the right image.
    right_boxes, right_detect_ms = timed_call(
        detect_boxes, model, right_rect, args.conf, args.imgsz, args.device
    )
    center_output, center_ms = timed_call(
        center_disparity, left_box, right_boxes, args.max_y_diff, args.max_disparity
    )
    center_disp, center_status = center_output
    methods["center_disparity"] = method_result(
        center_disp, center_status, right_detect_ms + center_ms,
        rectify_ms + left_detect_ms, calib, args
    )

    if center_status != "ok":
        # ROI-NCC does not require a right-image detector.  When no reliable
        # coarse stereo box pair is available, search the full physically
        # admissible disparity interval along the rectified epipolar band.
        # Keep SGBM and multi-patch GMM on the original coarse-guided protocol
        # so this fallback changes only the intended method.
        methods["roi_sgbm"]["status"] = "coarse_stereo_detection_failed"
        methods["multipatch_ncc_gmm"]["status"] = "coarse_stereo_detection_failed"
        ncc_output, ncc_ms = timed_call(
            roi_ncc_disparity, left_gray, right_gray, left_xyxy,
            args.max_disparity, args.y_tolerance, args.min_ncc_score,
            None, args.local_search_radius
        )
        ncc_disp, ncc_score, ncc_status = ncc_output
        methods["roi_ncc"] = method_result(
            ncc_disp, ncc_status, ncc_ms, rectify_ms + left_detect_ms,
            calib, args, ncc_score
        )
        return methods, left_box, rectify_ms, left_detect_ms, right_detect_ms

    sgbm_output, sgbm_ms = timed_call(
        roi_sgbm_disparity, left_gray, right_gray, left_xyxy,
        args.max_disparity, center_disp, args.local_search_radius
    )
    sgbm_disp, sgbm_status = sgbm_output
    methods["roi_sgbm"] = method_result(
        sgbm_disp, sgbm_status, sgbm_ms, rectify_ms + left_detect_ms, calib, args
    )

    ncc_output, ncc_ms = timed_call(
        roi_ncc_disparity, left_gray, right_gray, left_xyxy,
        args.max_disparity, args.y_tolerance, args.min_ncc_score,
        center_disp, args.local_search_radius
    )
    ncc_disp, ncc_score, ncc_status = ncc_output
    methods["roi_ncc"] = method_result(
        ncc_disp, ncc_status, ncc_ms, rectify_ms + left_detect_ms,
        calib, args, ncc_score
    )

    gmm_output, gmm_ms = timed_call(
        multipatch_ncc_gmm_disparity, left_gray, right_gray, left_xyxy,
        args.max_disparity, args.min_ncc_score,
        center_disp, args.local_search_radius
    )
    gmm_disp, gmm_score, gmm_status = gmm_output
    methods["multipatch_ncc_gmm"] = method_result(
        gmm_disp, gmm_status, gmm_ms, rectify_ms + left_detect_ms,
        calib, args, gmm_score
    )
    return methods, left_box, rectify_ms, left_detect_ms, right_detect_ms


def flatten_row(frame_meta, left_box, shared, methods):
    row = {
        "pair_index": int(frame_meta["pair_index"]),
        "source_frame_number": int(frame_meta["source_frame_number_1based"]),
        "video_time_ms": float(frame_meta["video_time_ms"]),
        "left_detection_conf": left_box["conf"] if left_box else None,
        "left_box_xyxy": json.dumps(left_box["xyxy"]) if left_box else "",
        **shared,
    }
    for method, result in methods.items():
        for key, value in result.items():
            row[f"{method}_{key}"] = value
    return row


def summarize(rows):
    summary = {}
    for method in METHODS:
        valid_depths = np.array([
            row[f"{method}_depth_m"] for row in rows
            if row[f"{method}_depth_m"] is not None
        ], dtype=float)
        total_times = np.array([
            row[f"{method}_total_ms"] for row in rows
            if row[f"{method}_total_ms"] is not None
        ], dtype=float)
        jumps = []
        for previous, current in zip(rows, rows[1:]):
            previous_depth = previous[f"{method}_depth_m"]
            current_depth = current[f"{method}_depth_m"]
            if (
                previous_depth is not None
                and current_depth is not None
                and current["source_frame_number"] - previous["source_frame_number"] == 1
            ):
                jumps.append(abs(current_depth - previous_depth) * 1000.0)
        jumps_mm = np.asarray(jumps, dtype=float)
        summary[method] = {
            "total_frames": len(rows),
            "valid_frames": int(valid_depths.size),
            "valid_rate": float(valid_depths.size / len(rows)) if rows else 0.0,
            "mean_depth_m": float(np.mean(valid_depths)) if valid_depths.size else None,
            "mean_adjacent_jump_mm": float(np.mean(jumps_mm)) if jumps_mm.size else None,
            "p95_adjacent_jump_mm": float(np.percentile(jumps_mm, 95)) if jumps_mm.size else None,
            "max_adjacent_jump_mm": float(np.max(jumps_mm)) if jumps_mm.size else None,
            "mean_total_ms": float(np.mean(total_times)) if total_times.size else None,
            "pipeline_fps": float(1000.0 / np.mean(total_times)) if total_times.size else None,
        }
    return summary


def save_plot(rows, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    labels = {
        "center_disparity": "Model F L/R center",
        "roi_sgbm": "Model F + ROI-SGBM",
        "roi_ncc": "Model F + ROI-NCC",
        "multipatch_ncc_gmm": "Model F + multi-NCC-GMM",
    }
    frame_numbers = [row["source_frame_number"] for row in rows]
    fig, axis = plt.subplots(figsize=(12, 6))
    for method in METHODS:
        depths = [row[f"{method}_depth_m"] for row in rows]
        axis.plot(frame_numbers, depths, marker="o", markersize=3, label=labels[method])
    axis.set_xlabel("Frame number")
    axis.set_ylabel("Estimated distance (m)")
    axis.set_title("Dynamic stereo matching comparison")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    path = output / "distance_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--output", type=Path, help="Default: <clip>/matching_comparison")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--max-disparity", type=int, default=256)
    parser.add_argument(
        "--local-search-radius", type=int, default=16,
        help="Fine matching range around Model F left/right coarse disparity (pixels)"
    )
    parser.add_argument("--max-y-diff", type=float, default=20.0)
    parser.add_argument(
        "--initial-point", nargs=2, type=float, metavar=("X", "Y"),
        help="Pixel point near the desired target in the first frame"
    )
    parser.add_argument(
        "--max-track-jump", type=float, default=200.0,
        help="Maximum target-center movement between adjacent frames (pixels)"
    )
    parser.add_argument("--y-tolerance", type=int, default=4)
    parser.add_argument("--min-ncc-score", type=float, default=0.55)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, help="Optional short test limit")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.clip.is_dir():
        raise NotADirectoryError(f"Clip directory not found: {args.clip}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model weight not found: {args.model}")
    if not args.calib.is_file():
        raise FileNotFoundError(f"Calibration file not found: {args.calib}")
    output = args.output or args.clip / "matching_comparison"
    output.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.clip)
    if args.max_frames:
        pairs = pairs[:args.max_frames]
    calib = load_calibration(args.calib)
    model = load_yolo_with_spdconv(str(args.model))

    # Warm up the detector before collecting timing data.
    _, warm_left_path, warm_right_path = pairs[0]
    warm_left = cv2.imread(str(warm_left_path))
    warm_right = cv2.imread(str(warm_right_path))
    warm_left, warm_right = rectify_pair(warm_left, warm_right, calib)
    print("Warming up Model F...")
    detect_boxes(model, warm_left, args.conf, args.imgsz, args.device)
    detect_boxes(model, warm_right, args.conf, args.imgsz, args.device)

    rows = []
    previous_left_box = None
    print(f"Model : {args.model}")
    print(f"Clip  : {args.clip}")
    print(f"Frames: {len(pairs)}")
    for index, (meta, left_path, right_path) in enumerate(pairs, 1):
        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))
        if left is None or right is None:
            raise RuntimeError(f"Unable to read pair: {left_path}, {right_path}")
        methods, left_box, rectify_ms, left_ms, right_ms = process_frame(
            model, left, right, calib, args, previous_left_box
        )
        if left_box is not None:
            previous_left_box = left_box
        row = flatten_row(meta, left_box, {
            "rectify_ms": rectify_ms,
            "left_detection_ms": left_ms,
            "right_detection_ms": right_ms,
        }, methods)
        rows.append(row)
        states = ", ".join(
            f"{name}={methods[name]['depth_m']:.3f}m"
            if methods[name]["depth_m"] is not None
            else f"{name}=FAIL({methods[name]['status']})"
            for name in METHODS
        )
        print(f"[{index:03d}/{len(pairs):03d}] Frame {row['source_frame_number']:06d}: {states}")

    csv_path = output / "per_frame_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    report = {
        "model_weight": str(args.model.resolve()),
        "calibration": str(args.calib.resolve()),
        "clip": str(args.clip.resolve()),
        "parameters": vars(args),
        "summary": summary,
    }
    report["parameters"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in report["parameters"].items()
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_path = save_plot(rows, output)

    print("\nComparison completed")
    for method, values in summary.items():
        mean_ms = values["mean_total_ms"]
        mean_ms_text = f"{mean_ms:.2f} ms" if mean_ms is not None else "N/A"
        print(
            f"  {method:22s}: valid={values['valid_frames']}/{values['total_frames']} "
            f"({values['valid_rate']:.1%}), mean_total={mean_ms_text}"
        )
    print(f"  Per-frame CSV: {csv_path}")
    print(f"  Summary JSON : {summary_path}")
    if plot_path:
        print(f"  Curve plot   : {plot_path}")


if __name__ == "__main__":
    main()

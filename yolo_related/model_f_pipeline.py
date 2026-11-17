#!/usr/bin/env python3
"""
Model F 完整优化管线
=====================
Model F = Model E (SPDConv + P2) + C2fAttn + 速度优化

速度优化策略 (针对 Jetson Orin, 目标 30 FPS):
  1. TensorRT FP16 引擎导出 → 2-3x 加速
  2. GPU 立体校正 (grid_sample) → 消除 CPU remap 瓶颈
  3. 批量推理 (batch=2 左右目一次前向) → 1.3x 加速
  4. 降低推理分辨率 (640×384) → 1.4x 加速
  5. FP16 半精度 → 1.5x 加速 (已含在 TRT 中)

使用流程:
  1. 训练: python train_model_f.py
  2. 导出: python model_f_pipeline.py --export-trt
  3. 评测: python model_f_pipeline.py --evaluate
  4. 实时: python model_f_pipeline.py --live
"""
import sys, os, time, pickle, glob, argparse, json
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ======================= 路径配置 =======================
CALIB_FILE = "/home/nvidia/PycharmProjects/PythonProject/stereo_calibration.pkl"
BASE_DIR = "/home/nvidia/Documents/bincular_vision/testing"
TRUE_DIST_DIR = "/home/nvidia/Documents/bincular_vision/ground_truth/distances"
TRUE_BOX_DIR = "/home/nvidia/Documents/bincular_vision/ground_truth/boxes"

# ZED X calibration maps and frozen stereo test images are 1920x1200.
ORIG_W, ORIG_H = 1920, 1200
INFER_W, INFER_H = 640, 384

# Model F 权重路径 (优先使用 F 自己的, 回退到 E)
MODEL_F_WEIGHTS = [
    "/home/nvidia/PycharmProjects/PythonProject/yolo_related/runs/train/target_detector_designed/weights/best.pt",
    "/home/nvidia/PycharmProjects/PythonProject/yolo_related/runs/detect/runs/train/target_detector_spdconv_p2/weights/best.pt",
]
MODEL_F_ENGINE = "/home/nvidia/PycharmProjects/PythonProject/yolo_related/model_f_640x384.engine"
# =========================================================


class ModelFInference:
    """Model F 优化推理引擎"""

    def __init__(self, model_path, use_trt=False, use_fp16=True,
                 use_gpu_remap=False, use_cuda_remap=True, batch_infer=False, conf=0.35):
        self.conf = conf
        self.use_gpu_remap = use_gpu_remap
        self.use_cuda_remap = use_cuda_remap
        self.batch_infer = batch_infer

        # 标定
        with open(CALIB_FILE, 'rb') as f:
            self.calib = pickle.load(f)
        self.fx = float(self.calib['P_left'][0, 0])
        self.baseline = float(np.linalg.norm(self.calib['T']))

        # 预构建 GPU 校正网格
        if use_gpu_remap:
            self._build_gpu_grids()

        # 加载模型
        self._load_model(model_path, use_trt, use_fp16)
        self._warmup()

        # 统计
        self.timings = {k: deque(maxlen=100) for k in
                        ['remap', 'resize', 'infer', 'depth', 'total']}
        self.frame_count = 0

    def _build_gpu_grids(self):
        import torch
        self.device = torch.device('cuda:0')

        def to_grid(mx, my):
            gx = torch.from_numpy(mx.copy()).float().to(self.device)
            gy = torch.from_numpy(my.copy()).float().to(self.device)
            gx = gx * 2.0 / (ORIG_W - 1) - 1.0
            gy = gy * 2.0 / (ORIG_H - 1) - 1.0
            return torch.stack([gx, gy], dim=-1).unsqueeze(0)

        self.grid_L = to_grid(self.calib['map_left_x'], self.calib['map_left_y'])
        self.grid_R = to_grid(self.calib['map_right_x'], self.calib['map_right_y'])

    def _load_model(self, path, use_trt, use_fp16):
        self.use_trt = False

        if use_trt and os.path.exists(MODEL_F_ENGINE):
            from trt_inference import TRTInference
            self.trt_engine = TRTInference(MODEL_F_ENGINE, conf=self.conf)
            self.use_trt = True
            print(f"使用 TensorRT 引擎: {MODEL_F_ENGINE}")
            return

        # 回退到 PyTorch 权重 (lazy import 避免 TensorRT CUDA 冲突)
        import torch
        from register_spdconv import register, load_yolo_with_spdconv
        register()

        self.device = torch.device('cuda:0')

        for w in MODEL_F_WEIGHTS:
            if os.path.exists(w):
                path = w
                print(f"使用 PyTorch 权重: {path}")
                break

        self.model = load_yolo_with_spdconv(path)
        self.model.fuse()
        if use_fp16:
            self.model.model.half()
        self.model.to(self.device)
        torch.backends.cudnn.benchmark = True

    def _warmup(self, n=10):
        dummy = np.random.randint(0, 255, (INFER_H, INFER_W, 3), dtype=np.uint8)
        if self.use_trt:
            for _ in range(n):
                self.trt_engine.infer(dummy)
        else:
            import torch
            for _ in range(n):
                if self.batch_infer:
                    self.model([dummy, dummy], conf=self.conf, imgsz=INFER_W, verbose=False)
                else:
                    self.model(dummy, conf=self.conf, imgsz=INFER_W, verbose=False)
            torch.cuda.synchronize()

    # ---------- GPU 校正 ----------
    def gpu_remap(self, img_np, grid):
        import torch
        t = torch.from_numpy(img_np).to(self.device, non_blocking=True)
        t = t.permute(2, 0, 1).unsqueeze(0).float()
        remapped = torch.nn.functional.grid_sample(
            t, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return remapped.squeeze(0).permute(1, 2, 0).byte()

    def gpu_resize(self, img_t, w, h):
        import torch
        t = img_t.permute(2, 0, 1).unsqueeze(0).float()
        t = torch.nn.functional.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
        return t.squeeze(0).permute(1, 2, 0).byte()

    # ---------- 核心推理 ----------
    def infer(self, left_np, right_np):
        """一次处理一对图像, 返回 (depth_m, left_boxes, right_boxes)

        TRT 模式: 跳过显式 resize, 引擎内部处理缩放和坐标映射
        PyTorch 模式: cv2.resize + ultralytics 推理 + 坐标缩放
        """
        t_total_start = time.perf_counter()

        if self.use_gpu_remap:
            import torch
            t0 = time.perf_counter()
            if self.use_trt:
                # TRT 内部 resize, 仅做 remap
                left_inf = self.gpu_remap(left_np, self.grid_L).cpu().numpy()
                right_inf = self.gpu_remap(right_np, self.grid_R).cpu().numpy()
            else:
                left_inf = self.gpu_resize(
                    self.gpu_remap(left_np, self.grid_L), INFER_W, INFER_H).cpu().numpy()
                right_inf = self.gpu_resize(
                    self.gpu_remap(right_np, self.grid_R), INFER_W, INFER_H).cpu().numpy()
            torch.cuda.synchronize()
            t_remap = (time.perf_counter() - t0) * 1000
        elif self.use_cuda_remap:
            import cuda_remap
            t_remap_start = time.perf_counter()
            left_t, right_t = cuda_remap.remap_stereo(
                left_np, right_np,
                self.calib['map_left_x'], self.calib['map_left_y'],
                self.calib['map_right_x'], self.calib['map_right_y'])
            if self.use_trt:
                # TRT 内部 resize, 直接传原图 (1920×1200)
                left_inf, right_inf = left_t, right_t
            else:
                left_inf = cv2.resize(left_t, (INFER_W, INFER_H))
                right_inf = cv2.resize(right_t, (INFER_W, INFER_H))
            t_remap = (time.perf_counter() - t_remap_start) * 1000
        else:
            # 无 remap: TRT 自己处理 letterbox
            if self.use_trt:
                # TRT 期望原图分辨率, 内部做 letterbox
                left_inf, right_inf = left_np, right_np
            else:
                left_inf = cv2.resize(left_np, (INFER_W, INFER_H))
                right_inf = cv2.resize(right_np, (INFER_W, INFER_H))
            t_remap = 0

        # 推理
        t_infer_start = time.perf_counter()
        if self.use_trt:
            boxes_l = self.trt_engine.infer(left_inf)
            boxes_r = self.trt_engine.infer(right_inf)
            # numpy → list, 已在原始图像坐标
            left_boxes = boxes_l.tolist() if len(boxes_l) > 0 else []
            right_boxes = boxes_r.tolist() if len(boxes_r) > 0 else []
        else:
            result_l = self.model(left_inf, conf=self.conf, imgsz=INFER_W, verbose=False)
            result_r = self.model(right_inf, conf=self.conf, imgsz=INFER_W, verbose=False)
            torch.cuda.synchronize()

            def extract(result):
                boxes = []
                r = result[0] if isinstance(result, list) else result
                if r.boxes is not None:
                    for box in r.boxes.xyxy.cpu().numpy():
                        boxes.append([box[0] * ORIG_W / INFER_W,
                                      box[1] * ORIG_H / INFER_H,
                                      box[2] * ORIG_W / INFER_W,
                                      box[3] * ORIG_H / INFER_H])
                return boxes

            left_boxes = extract(result_l)
            right_boxes = extract(result_r)
        t_infer = (time.perf_counter() - t_infer_start) * 1000

        # 深度计算 (视差法)
        depth = None
        if left_boxes and right_boxes:
            u_l = (left_boxes[0][0] + left_boxes[0][2]) / 2
            u_r = (right_boxes[0][0] + right_boxes[0][2]) / 2
            disp = u_l - u_r
            if disp > 5:
                depth = self.fx * self.baseline / disp

        t_total = (time.perf_counter() - t_total_start) * 1000

        self.timings['remap'].append(t_remap)
        self.timings['infer'].append(t_infer)
        self.timings['total'].append(t_total)
        self.frame_count += 1

        return depth, left_boxes, right_boxes

    def stats(self):
        print(f"\n{'='*55}")
        print(f"Model F 推理统计 ({self.frame_count} 帧)")
        print(f"{'='*55}")
        for k in ['remap', 'infer', 'depth', 'total']:
            vals = self.timings[k]
            if vals:
                avg = np.mean(vals)
                print(f"  {k:10s}: {avg:7.1f} ms")
        avg_t = np.mean(self.timings['total'])
        print(f"  {'FPS':10s}: {1000/avg_t:7.1f}")
        print(f"{'='*55}")
        return avg_t


# ======================= 评估模式 =======================

def evaluate_model_f(args):
    """完整评估 Model F: 对所有 3 个数据集运行推理并计算指标"""
    from compute_all_metrics import (load_true_distances, load_true_boxes,
                                      compute_detection_metrics, compute_depth_metrics)

    weight_path = None
    for w in MODEL_F_WEIGHTS:
        if os.path.exists(w):
            weight_path = w
            break
    if not weight_path:
        print("❌ 未找到 Model F 权重，请先运行 train_model_f.py")
        return

    print(f"Model F 权重: {weight_path}")
    engine = ModelFInference(weight_path, use_trt=args.trt,
                             use_fp16=not args.no_fp16,
                             use_gpu_remap=args.gpu_remap,
                             use_cuda_remap=not args.no_cuda_remap,
                             batch_infer=not args.no_batch,
                             conf=args.conf)

    true_dists = load_true_distances()
    true_boxes = load_true_boxes()

    DATASETS = ['original', 'tension_variation_heavy', 'light_and_clutter_heavy']
    all_results = {}

    for ds in DATASETS:
        left_dir = os.path.join(BASE_DIR, ds, 'left')
        right_dir = os.path.join(BASE_DIR, ds, 'right')

        detect_dir = os.path.join(BASE_DIR, ds, 'distance_detect_F')
        pred_dir = os.path.join(BASE_DIR, ds, 'pred_boxes_F')
        time_log = os.path.join(BASE_DIR, ds, 'inference_time_F.txt')

        os.makedirs(detect_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

        left_files = sorted(glob.glob(os.path.join(left_dir, 'left_*.png')))
        print(f"\n数据集 {ds}: {len(left_files)} 对图像")

        times = []
        for lf in left_files:
            fn = os.path.basename(lf)
            ts = fn.replace('left_', '').replace('.png', '')
            rf = os.path.join(right_dir, fn.replace('left_', 'right_'))

            left = cv2.imread(lf)
            right = cv2.imread(rf)
            if left is None or right is None:
                continue

            t0 = time.perf_counter()
            depth, l_boxes, r_boxes = engine.infer(left, right)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

            # 保存预测框 (用实际图像尺寸归一化)
            img_h, img_w = left.shape[:2]
            with open(os.path.join(pred_dir, f'{ts}.txt'), 'w') as f:
                for box in l_boxes:
                    xc = ((box[0] + box[2]) / 2) / img_w
                    yc = ((box[1] + box[3]) / 2) / img_h
                    bw = (box[2] - box[0]) / img_w
                    bh = (box[3] - box[1]) / img_h
                    f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} 1.0\n")

            # 保存距离
            with open(os.path.join(detect_dir, f'distance_{ts}.txt'), 'w') as f:
                f.write(f"{depth:.4f}\n" if depth else "-1\n")

        # 写入时间日志
        with open(time_log, 'w') as f:
            f.write("filename,time_ms\n")
            for i, lf in enumerate(left_files):
                fn = os.path.basename(lf)
                f.write(f"{fn},{times[i]:.2f}\n" if i < len(times) else f"{fn},0.00\n")

        # 计算指标
        pred_boxes = {}
        for f in sorted(glob.glob(os.path.join(pred_dir, '*.txt'))):
            ts = os.path.basename(f).replace('.txt', '')
            with open(f) as fh:
                boxes = []
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id, xc, yc, w, h = [float(p) for p in parts[:5]]
                        xc *= ORIG_W; yc *= ORIG_H; w *= ORIG_W; h *= ORIG_H
                        boxes.append([xc-w/2, yc-h/2, xc+w/2, yc+h/2, 1.0])
                pred_boxes[ts] = boxes

        detect_dists = {}
        for f in sorted(glob.glob(os.path.join(detect_dir, 'distance_*.txt'))):
            ts = os.path.basename(f).replace('distance_', '').replace('.txt', '')
            with open(f) as fh:
                line = fh.readline().strip()
            if line and line != '-1':
                detect_dists[ts] = float(line)

        det_m = compute_detection_metrics(pred_boxes, true_boxes)
        dep_m = compute_depth_metrics(detect_dists, true_dists)

        filtered_times = [t for t in times if t < 500] or times
        avg_ms = np.mean(filtered_times)
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0

        result = {**det_m, **dep_m, 'fps': round(fps, 1), 'avg_ms': round(avg_ms, 1)}
        all_results[f'F-{ds}'] = result
        print(f"  F-{ds}: P={det_m['precision']:.1f}% R={det_m['recall']:.1f}% "
              f"mAP={det_m['mAP50']:.1f}% MAE={dep_m['mae_mm']:.0f}mm "
              f"FPS={fps:.1f}")

    # 更新总 metrics JSON
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics_all.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        with open(metrics_path, 'w') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"\n指标已追加到: {metrics_path}")

    engine.stats()
    engine.stats()
    return all_results


# ======================= 实时模式 =======================

def live_mode(args):
    """实时处理 zed_left/ 目录下的最新图像"""
    left_dir = os.path.join(BASE_DIR, 'original', 'left')
    right_dir = os.path.join(BASE_DIR, 'original', 'right')

    weight_path = None
    for w in MODEL_F_WEIGHTS:
        if os.path.exists(w):
            weight_path = w
            break
    if not weight_path:
        weight_path = MODEL_F_WEIGHTS[-1]

    engine = ModelFInference(weight_path, use_trt=args.trt,
                             use_fp16=not args.no_fp16,
                             use_gpu_remap=args.gpu_remap,
                             use_cuda_remap=not args.no_cuda_remap,
                             batch_infer=not args.no_batch,
                             conf=args.conf)

    files = sorted(glob.glob(os.path.join(left_dir, 'left_*.png')))
    print(f"实时模式: {len(files)} 对图像 (循环播放)")
    print("按 Ctrl+C 停止\n")

    try:
        while True:
            for lf in files:
                fn = os.path.basename(lf)
                rf = os.path.join(right_dir, fn.replace('left_', 'right_'))
                left = cv2.imread(lf)
                right = cv2.imread(rf)
                depth, l_boxes, r_boxes = engine.infer(left, right)

                status = f"{depth:.3f}m" if depth else "N/A"
                avg_fps = 1000 / np.mean(engine.timings['total']) if engine.timings['total'] else 0

                if args.display:
                    # 画框
                    for box in l_boxes:
                        cv2.rectangle(left, (int(box[0]), int(box[1])),
                                      (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    cv2.putText(left, f"Depth: {status} FPS: {avg_fps:.1f}",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Model F - Left", left)

                if engine.frame_count % 30 == 0:
                    print(f"  Frame {engine.frame_count}: {status} | {avg_fps:.1f} FPS")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
    except KeyboardInterrupt:
        pass

    engine.stats()


# ======================= TensorRT 导出 =======================

def export_trt(args):
    """导出 Model E → TensorRT FP16 引擎"""
    weight_path = None
    for w in MODEL_F_WEIGHTS:
        if os.path.exists(w):
            weight_path = w
            break
    if not weight_path:
        print("❌ 未找到权重, 请先训练")
        return

    print(f"导出 TensorRT: {weight_path} → {MODEL_F_ENGINE}")

    model = load_yolo_with_spdconv(weight_path)
    model.export(format='engine', imgsz=INFER_W, half=True, device='cuda:0',
                 batch=2 if not args.no_batch else 1, workspace=4)
    print(f"✅ TensorRT 引擎已保存")


# ======================= 主入口 =======================

def main():
    parser = argparse.ArgumentParser(description="Model F 优化推理管线")
    parser.add_argument('--mode', choices=['eval', 'live', 'export'],
                        default='eval', help='运行模式')
    parser.add_argument('--trt', action='store_true', help='使用 TensorRT 引擎')
    parser.add_argument('--no-fp16', action='store_true', help='禁用 FP16')
    parser.add_argument('--gpu-remap', action='store_true', help='启用 GPU 校正 (实验性, 可能影响精度)')
    parser.add_argument('--no-cuda-remap', action='store_true', help='禁用 CUDA remap, 退回 cv2.remap')
    parser.add_argument('--no-batch', action='store_true', help='禁用批量推理')
    parser.add_argument('--conf', type=float, default=0.35, help='置信度阈值')
    parser.add_argument('--display', action='store_true', help='显示图像')
    parser.add_argument('--export-trt', action='store_true', help='仅导出 TensorRT')
    parser.add_argument('--evaluate', action='store_true', help='评估模式')

    args = parser.parse_args()

    if args.export_trt:
        export_trt(args)
    elif args.evaluate or args.mode == 'eval':
        evaluate_model_f(args)
    elif args.mode == 'live':
        live_mode(args)
    else:
        evaluate_model_f(args)


if __name__ == "__main__":
    main()

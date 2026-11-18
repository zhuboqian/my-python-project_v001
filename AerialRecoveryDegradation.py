#!/usr/bin/env python3
"""
空中回收场景数据集退化处理 - 去除湍流，适配新路径
无气动光学湍流，保留：运动模糊、光照变化、背景杂波、传感器噪声，均可独立开关。
"""

import os
import cv2
import numpy as np
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DegradationSimulator:
    """退化模拟器（无湍流）"""

    def __init__(self, tile_size: int = 640, focal_length: float = 800.0):
        self.tile_size = tile_size
        self.focal_length = focal_length
        self.baseline = 0.12
        self.exposure_time = 0.005
        self.pixel_size = 3.45e-6
        print(f"[退化器] 焦距={focal_length}px")

    # ------------------ 非线性运动模糊 ------------------
    def nonlinear_dynamic_blur_kernel(self, size: int,
                                      tension_variation: float = 0.15,
                                      flight_speed: float = 60.0,
                                      relative_distance: float = 5.0) -> np.ndarray:
        psf = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        damping_ratio = max(0.15, 0.65 - tension_variation * 0.8)
        natural_freq = 12.0
        max_accel = 9.2 + tension_variation * 15.0
        pixel_motion = (max_accel * self.exposure_time ** 2 * self.focal_length /
                        (relative_distance * self.pixel_size * 1000))
        pixel_scale = np.clip(pixel_motion * 8.0, 20, 55)
        time_scale = 60.0
        t_physical = np.linspace(0, self.exposure_time * time_scale, 200)
        wd = natural_freq * np.sqrt(max(0, 1 - damping_ratio ** 2))

        if wd > 0:
            envelope = np.exp(-damping_ratio * natural_freq * t_physical)
            term1 = np.cos(wd * t_physical)
            term2 = (damping_ratio / np.sqrt(1 - damping_ratio ** 2)) * np.sin(wd * t_physical)
            response = 1.0 - envelope * (term1 + term2)
        else:
            response = 1.0 - np.exp(-natural_freq * t_physical)

        wind_angle = random.uniform(-15, 15)
        angle_rad = np.deg2rad(wind_angle)
        main_disp = response * pixel_scale
        cross_disp = 0.3 * np.sin(wd * t_physical) * np.exp(-damping_ratio * natural_freq * t_physical) * pixel_scale

        num_samples = 80
        indices = np.linspace(0, len(t_physical) - 1, num_samples, dtype=int)
        for idx in indices:
            x_rot = main_disp[idx] * np.cos(angle_rad) - cross_disp[idx] * np.sin(angle_rad)
            y_rot = main_disp[idx] * np.sin(angle_rad) + cross_disp[idx] * np.cos(angle_rad)
            x = int(center + x_rot)
            y = int(center + y_rot)
            if 0 <= x < size and 0 <= y < size:
                psf[y, x] += 1.0

        psf = cv2.GaussianBlur(psf, (5, 5), 1.0)
        if psf.sum() > 0:
            psf /= psf.sum()
        else:
            psf[center, center] = 1.0
        return psf

    # ------------------ 光照变化（含过曝、渐晕）------------------
    def lighting_variation(self, image: np.ndarray,
                           strength: float = 0.3,
                           overexpose_target: bool = False,
                           target_bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        img_float = image.astype(np.float32) / 255.0
        brightness_factor = 1.0 + random.uniform(-0.4, 0.6) * strength
        contrast_factor = 1.0 + random.uniform(-0.2, 0.4) * strength
        img_float = contrast_factor * img_float + (brightness_factor - 1.0)

        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        vignette = 1.0 - 0.3 * strength * ((x - cx) ** 2 + (y - cy) ** 2) / (cx ** 2 + cy ** 2)
        vignette = np.clip(vignette, 0.7, 1.0)
        if len(image.shape) == 3:
            vignette = np.dstack([vignette] * 3)
        img_float = img_float * vignette

        if overexpose_target:
            if target_bbox is None:
                bw = random.randint(int(w * 0.05), int(w * 0.15))
                bh = random.randint(int(h * 0.05), int(h * 0.15))
                bx = random.randint(0, w - bw)
                by = random.randint(0, h - bh)
                target_bbox = (bx, by, bx + bw, by + bh)
            x1, y1, x2, y2 = target_bbox
            mask = np.zeros((h, w), dtype=np.float32)
            cy_box = (y1 + y2) // 2
            cx_box = (x1 + x2) // 2
            sigma = max((x2 - x1), (y2 - y1)) * 0.5
            Y, X = np.ogrid[:h, :w]
            gaussian = np.exp(-((X - cx_box) ** 2 + (Y - cy_box) ** 2) / (2 * sigma ** 2))
            mask = gaussian * 0.8 * strength
            if len(image.shape) == 3:
                img_float += np.dstack([mask] * 3)
            else:
                img_float += mask

        img_float = np.clip(img_float, 0.0, 1.0)
        return (img_float * 255.0).astype(np.uint8)

    # ------------------ 背景杂波 ------------------
    def background_clutter(self, image: np.ndarray, strength: float = 0.2) -> np.ndarray:
        h, w = image.shape[:2]
        noise = np.zeros((h, w), dtype=np.float32)
        for scale in [64, 32, 16]:
            noise_small = np.random.randn(h // scale + 1, w // scale + 1).astype(np.float32)
            noise_up = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_LINEAR)
            noise += noise_up
        noise = (noise - noise.mean()) / (noise.std() + 1e-6)
        noise = noise * strength * 40
        img_float = image.astype(np.float32)
        if len(image.shape) == 3:
            noise_3ch = np.dstack([noise] * 3)
            img_float += noise_3ch
        else:
            img_float += noise
        return np.clip(img_float, 0, 255).astype(np.uint8)

    # ------------------ 传感器噪声 ------------------
    def add_physical_noise(self, image: np.ndarray, iso_level: float = 400.0) -> np.ndarray:
        img_float = image.astype(np.float32) / 255.0
        photons = np.random.poisson(img_float * iso_level) / iso_level
        read_noise = np.random.normal(0, 0.005 * (iso_level / 100.0), image.shape)
        result = photons + read_noise
        return np.clip(result * 255.0, 0, 255).astype(np.uint8)

    # ------------------ 完整退化（无湍流）------------------
    def apply_degradation(self, image: np.ndarray,
                          flight_speed: float = 60.0,
                          relative_distance: float = 5.0,
                          tension_variation: float = 0.15,
                          enable_lighting: bool = True,
                          lighting_strength: float = 0.3,
                          overexpose_target: bool = False,
                          enable_clutter: bool = True,
                          clutter_strength: float = 0.2,
                          enable_noise: bool = True,
                          target_bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:  # 新增参数
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # 1. 运动模糊
        kernel_size = min(25, max(11, int(15 + tension_variation * 15)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        psf = self.nonlinear_dynamic_blur_kernel(kernel_size, tension_variation, flight_speed, relative_distance)
        image = cv2.filter2D(image, -1, psf)

        # 2. 光照变化（传入靶标区域）
        if enable_lighting and lighting_strength > 0.01:
            image = self.lighting_variation(image, strength=lighting_strength,
                                            overexpose_target=overexpose_target,
                                            target_bbox=target_bbox)  # 传递固定区域

        # 3. 背景杂波
        if enable_clutter and clutter_strength > 0.01:
            image = self.background_clutter(image, strength=clutter_strength)

        # 4. 传感器噪声
        if enable_noise:
            image = self.add_physical_noise(image, iso_level=400 + tension_variation * 800)

        return image


class StereoDatasetProcessor:
    """处理左右图像对，适配新目录结构"""

    def load_target_bbox(self,image_path: Path, img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int]]:
        """从同名 .txt 标注文件读取第一个目标的边界框（YOLO 归一化格式 -> 像素坐标）"""
        label_path = image_path.with_suffix('.txt')  # 假设图像和标注在同一文件夹
        if not label_path.exists():
            return None
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, cx, cy, w_box, h_box = map(float, parts[:5])
                    x1 = int((cx - w_box / 2) * img_w)
                    y1 = int((cy - h_box / 2) * img_h)
                    x2 = int((cx + w_box / 2) * img_w)
                    y2 = int((cy + h_box / 2) * img_h)
                    return (x1, y1, x2, y2)  # 返回第一个靶标区域
        return None

    def __init__(self,
                 input_base: str,
                 output_base: str,
                 degradation_level: str = 'heavy'):
        self.input_base = Path(input_base)
        self.output_base = Path(output_base)
        self.left_input_dir = self.input_base / 'left_distance'/'0-3m'
        self.right_input_dir = self.input_base / 'right_distance'/'0-3m'

        if not self.left_input_dir.exists() or not self.right_input_dir.exists():
            raise FileNotFoundError("需要 left_distance 和 right_distance 目录")

        # 输出路径：Processed_image/1/
        self.output_left_dir = self.output_base / 'Processed_image' / 'light_and_clutter_heavy' / 'left'
        self.output_right_dir = self.output_base / 'Processed_image' / 'light_and_clutter_heavy'/ 'right'
        self.output_left_dir.mkdir(parents=True, exist_ok=True)
        self.output_right_dir.mkdir(parents=True, exist_ok=True)

        self.degradator = DegradationSimulator(tile_size=512)

        # 退化等级预设（无湍流）
        self.level_params = {
            'light': {
                'tension_variation': 0,
                'enable_lighting': True,
                'lighting_strength': 0.15,
                'overexpose_target': True,
                'enable_clutter':True,
                'clutter_strength': 0.1
            },
            'medium': {
                'tension_variation': 0,
                'enable_lighting': True,
                'lighting_strength': 0.3,
                'overexpose_target': True,
                'enable_clutter': True,
                'clutter_strength': 0.2
            },
            'heavy': {
                'tension_variation': 0,
                'enable_lighting': True,
                'lighting_strength': 0.5,
                'overexpose_target': True,
                'enable_clutter':True,
                'clutter_strength': 0.35
            },

        }
        self.level = degradation_level
        self.params = self.level_params[degradation_level]

        self.processed = 0
        self.failed = []

    def _find_right_image(self, left_path: Path) -> Optional[Path]:
        """根据左图路径查找对应右图（将 left_ 前缀替换为 right_）"""
        # 假设左图文件名以 left_ 开头，右图以 right_ 开头
        left_name = left_path.name
        if left_name.startswith('left_'):
            right_name = left_name.replace('left_', 'right_', 1)
        else:
            # 如果没有标准前缀，尝试直接匹配相同文件名
            right_name = left_name
        # 在 right_distance 的对应子目录中查找
        # 保留相对子路径（如 0-3m）
        rel_subdir = left_path.parent.relative_to(self.left_input_dir)
        right_path = self.right_input_dir / rel_subdir / right_name
        if right_path.exists():
            return right_path
        # 如果不成功，尝试全局搜索同名文件（可能子目录结构不同）
        for p in self.right_input_dir.rglob(right_name):
            return p
        return None

    def process_all(self):
        # 收集所有左图
        left_images = list(self.left_input_dir.rglob('*.png')) + \
                      list(self.left_input_dir.rglob('*.jpg'))
        print(f"找到 {len(left_images)} 张左图")

        for left_path in tqdm(left_images, desc="处理中"):
            right_path = self._find_right_image(left_path)
            if right_path is None:
                self.failed.append({'file': str(left_path), 'error': '无对应右图'})
                continue

            try:
                left_img = cv2.imread(str(left_path))
                right_img = cv2.imread(str(right_path))
                if left_img is None or right_img is None:
                    raise ValueError("无法读取图像")

                h, w = left_img.shape[:2]  # 假设左右图尺寸相同

                # 读取靶标边界框（左图和右图各自读取）
                left_bbox = self.load_target_bbox(left_path, w, h)
                right_bbox = self.load_target_bbox(right_path, w, h)  # 注意右图标注文件名与右图对应

                speed = random.uniform(55, 70)
                dist = random.uniform(3.5, 6.5)

                left_degraded = self.degradator.apply_degradation(
                    left_img, flight_speed=speed, relative_distance=dist,
                    target_bbox=left_bbox, **self.params)
                right_degraded = self.degradator.apply_degradation(
                    right_img, flight_speed=speed, relative_distance=dist,
                    target_bbox=right_bbox, **self.params)
                # 保存，文件名保持原左图名（可以同时保留右图原名）
                output_left_path = self.output_left_dir / left_path.name
                output_right_path = self.output_right_dir / right_path.name
                cv2.imwrite(str(output_left_path), left_degraded)
                cv2.imwrite(str(output_right_path), right_degraded)
                self.processed += 1
            except Exception as e:
                self.failed.append({'file': str(left_path), 'error': str(e)})
                print(f"\n❌ 失败: {left_path.name} - {str(e)}")

        # 输出报告
        print(f"\n完成: 成功 {self.processed} 对, 失败 {len(self.failed)} 对")
        report = {
            'input_base': str(self.input_base),
            'output_base': str(self.output_base),
            'level': self.level,
            'processed': self.processed,
            'failed': len(self.failed)
        }
        with open(self.output_base / 'Processed_image' /'tension_variation_heavy'/ 'degradation_report.json', 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_base', type=str,
                        default='/home/nvidia/Documents/bincular_vision/photos',
                        help='原始数据集根目录（包含 left_distance 和 right_distance）')
    parser.add_argument('--output_base', type=str,
                        default='/home/nvidia/Documents/bincular_vision/photos',
                        help='输出根目录，将在其下创建 Processed_image/1/left 和 right')
    parser.add_argument('--level', type=str, default='heavy',
                        choices=['light', 'medium', 'heavy'])
    args = parser.parse_args()

    processor = StereoDatasetProcessor(args.input_base, args.output_base, args.level)
    processor.process_all()
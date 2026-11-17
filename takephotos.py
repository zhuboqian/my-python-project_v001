import pyzed.sl as sl
import cv2
import numpy as np
import os
import argparse
from datetime import datetime
import glob


class ZEDCamera:
    """ZED相机基础类 - 适配 ZED X"""

    def __init__(self, resolution=sl.RESOLUTION.HD1200, fps=15):
        """
        初始化ZED相机
        :param resolution: 分辨率（默认使用 HD1200）
        :param fps: 帧率（根据测试，15fps 工作正常）
        """
        self.zed = sl.Camera()

        # 获取相机列表
        cameras = sl.Camera.get_device_list()
        if len(cameras) == 0:
            raise RuntimeError("未检测到ZED相机")

        print(f"检测到 {len(cameras)} 个ZED相机")
        for i, cam in enumerate(cameras):
            print(f"  相机 {i}: 序列号 {cam.serial_number}")

        # 设置初始化参数
        self.init_params = sl.InitParameters()

        # 如果有多个相机，使用第一个
        if len(cameras) > 0:
            self.init_params.set_from_serial_number(cameras[0].serial_number)

        # 设置分辨率和帧率
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.NONE  # 拍照不需要深度

        # 打开相机
        print(f"\n尝试打开相机 (分辨率: {resolution}, 帧率: {fps})...")
        err = self.zed.open(self.init_params)

        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"相机打开失败: {err}")

        # 获取相机信息
        camera_info = self.zed.get_camera_information()

        # 安全地获取相机信息（避免属性不存在错误）
        camera_model = getattr(camera_info, 'camera_model', 'Unknown')
        serial_number = getattr(camera_info, 'serial_number', 'Unknown')

        # 固件版本可能在不同位置
        firmware_version = 'Unknown'
        try:
            if hasattr(camera_info, 'firmware_version'):
                firmware_version = camera_info.firmware_version
            elif hasattr(camera_info, 'sensors_configuration'):
                if hasattr(camera_info.sensors_configuration, 'firmware_version'):
                    firmware_version = camera_info.sensors_configuration.firmware_version
        except:
            pass

        # 获取实际分辨率（通过抓取一帧）
        print("获取图像分辨率...")
        test_image = sl.Mat()
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(test_image, sl.VIEW.LEFT)
            img_data = test_image.get_data()
            if img_data is not None:
                self.height, self.width = img_data.shape[:2]
            else:
                self.width, self.height = 1920, 1200  # 默认值
        else:
            self.width, self.height = 1920, 1200  # 默认值

        print(f"\n✓ 相机打开成功")
        print(f"  相机型号: {camera_model}")
        print(f"  序列号: {serial_number}")
        print(f"  固件版本: {firmware_version}")
        print(f"  分辨率: {self.width} x {self.height}")
        print(f"  帧率: {fps} FPS")

        # 创建图像容器
        self.image_left = sl.Mat()
        self.image_right = sl.Mat()

    def capture_left(self):
        """捕获左目图像"""
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            return self.image_left.get_data()
        return None

    def capture_right(self):
        """捕获右目图像"""
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
            return self.image_right.get_data()
        return None

    def capture_stereo(self):
        """捕获左右目图像对"""
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
            return self.image_left.get_data(), self.image_right.get_data()
        return None, None

    def save_image(self, image, filename):
        """保存图像"""
        cv2.imwrite(filename, image)
        print(f"已保存: {filename}")

    def close(self):
        """关闭相机"""
        self.zed.close()
        print("相机已关闭")


def parse_args():
    parser = argparse.ArgumentParser(description="ZED X 多靶标双目图像采集")
    parser.add_argument(
        "--output",
        default="/home/nvidia/Documents/bincular_vision/multi_target_data/raw",
        help="数据集根目录；程序将在其中创建 left 和 right 子目录",
    )
    return parser.parse_args()


def main():
    """主函数 - 纯拍照模式（不输入距离）"""

    args = parse_args()

    # 创建相机对象
    try:
        # 使用 HD1200 分辨率，15fps
        camera = ZEDCamera(resolution=sl.RESOLUTION.HD1200, fps=15)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # 多靶标数据与原有单靶标数据分开存放，左右目也必须分目录保存。
    # 先保存在 raw 下，完成标注后再划分训练集和冻结测试集。
    base_dir = os.path.abspath(os.path.expanduser(args.output))
    left_dir = os.path.join(base_dir, "left")
    right_dir = os.path.join(base_dir, "right")

    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    print(f"\n图像保存路径:")
    print(f"  左目图像: {left_dir}")
    print(f"  右目图像: {right_dir}")

    print("\n" + "=" * 50)
    print("ZED X 相机拍照程序（纯拍照模式）")
    print("=" * 50)
    print("操作说明:")
    print("  [s] - 仅保存左目图像")
    print("  [d] - 仅保存右目图像")
    print("  [b] - 保存同步双目图像对（多靶标采集建议使用）")
    print("  [p] - 预览模式（不保存）")
    print("  [q] - 退出")
    print("=" * 50)

    photo_count = 0
    preview_mode = False

    # 获取已有图像数量（用于继续编号）
    existing_left = len(glob.glob(os.path.join(left_dir, "*.png")))
    existing_right = len(glob.glob(os.path.join(right_dir, "*.png")))
    photo_count = max(existing_left, existing_right)
    print(f"\n已有图像数量: {photo_count} 张")

    try:
        while True:
            # 获取立体图像
            left_img, right_img = camera.capture_stereo()
            if left_img is None or right_img is None:
                print("获取图像失败，重试...")
                continue

            # 创建显示图像
            if preview_mode:
                # 预览模式：显示左右目并排
                h, w = left_img.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    new_w = 1280
                    new_h = int(h * scale)
                    left_display = cv2.resize(left_img, (new_w, new_h))
                    right_display = cv2.resize(right_img, (new_w, new_h))
                    combined = np.hstack((left_display, right_display))
                else:
                    combined = np.hstack((left_img, right_img))

                cv2.putText(combined, "PREVIEW MODE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("ZED X - PREVIEW MODE", combined)
            else:
                # 正常模式：显示左目图像
                display_img = left_img.copy()

                # 添加文字信息
                cv2.putText(display_img, f"Photos: {photo_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_img, "S:Left D:Right B:Both P:Preview Q:Quit",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # 显示分辨率信息和保存路径
                h, w = display_img.shape[:2]
                cv2.putText(display_img, f"{w}x{h}", (w - 100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_img, f"Save to: {base_dir}", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("ZED X Camera", display_img)

            # 处理按键
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):          # 保存左目（不输入距离）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(left_dir, f"left_{timestamp}_{photo_count:04d}.png")
                camera.save_image(left_img, filename)
                photo_count += 1

            elif key == ord('d'):        # 保存右目
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(right_dir, f"right_{timestamp}_{photo_count:04d}.png")
                camera.save_image(right_img, filename)
                photo_count += 1

            elif key == ord('b'):        # 保存双目对
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                left_filename = os.path.join(left_dir, f"left_{timestamp}_{photo_count:04d}.png")
                right_filename = os.path.join(right_dir, f"right_{timestamp}_{photo_count:04d}.png")
                camera.save_image(left_img, left_filename)
                camera.save_image(right_img, right_filename)
                print(f"已保存双目图像对 #{photo_count}")
                photo_count += 1

            elif key == ord('p'):        # 切换预览模式
                preview_mode = not preview_mode
                mode_text = "预览" if preview_mode else "正常"
                print(f"切换到{mode_text}模式")

            elif key == ord('q'):        # 退出
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        camera.close()
        cv2.destroyAllWindows()
        print(f"\n拍照完成，共保存 {photo_count} 张照片")
        print(f"左目图像保存在: {left_dir}")
        print(f"右目图像保存在: {right_dir}")


if __name__ == "__main__":
    main()

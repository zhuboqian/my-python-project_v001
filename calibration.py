import numpy as np
import cv2
import glob
import os
import pickle


class ZedXStereoCalibration:
    """ZED X 相机双目标定类"""

    def __init__(self, chessboard_size=(11, 8), square_size=0.04):
        """
        初始化标定器
        :param chessboard_size: 棋盘格内角点数 (宽度方向角点数, 高度方向角点数)
                              宽度方向: 11个内角点
                              高度方向: 8个内角点
        :param square_size: 棋盘格方格边长（米），40mm = 0.04米
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size

        # 亚像素角点优化终止条件
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 准备世界坐标系中的3D点 (假设棋盘格在Z=0平面上)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size  # 转换为实际物理尺寸

        # 存储点集
        self.objpoints = []  # 3D世界坐标点
        self.imgpoints_left = []  # 左相机2D图像点
        self.imgpoints_right = []  # 右相机2D图像点

        # 图像尺寸
        self.image_size = None

    def load_images(self, left_dir, right_dir):
        """
        加载左右相机图像并检测棋盘格角点
        :param left_dir: 左相机图像目录
        :param right_dir: 右相机图像目录
        :return: 有效图像对数量
        """
        # 获取所有图像文件
        left_images = sorted(glob.glob(os.path.join(left_dir, '*.png')))
        right_images = sorted(glob.glob(os.path.join(right_dir, '*.png')))

        if len(left_images) == 0 or len(right_images) == 0:
            print(f"错误: 未找到图像文件")
            print(f"左图像目录: {left_dir}")
            print(f"右图像目录: {right_dir}")
            return 0

        # 确保左右图像数量匹配
        min_count = min(len(left_images), len(right_images))
        left_images = left_images[:min_count]
        right_images = right_images[:min_count]

        print(f"找到 {len(left_images)} 对图像")
        print(f"棋盘格规格: {self.chessboard_size[0]}x{self.chessboard_size[1]} 内角点")
        print(
            f"棋盘格实际尺寸: {self.chessboard_size[0] * self.square_size * 1000:.0f} x {self.chessboard_size[1] * self.square_size * 1000:.0f} mm")
        print(f"方格边长: {self.square_size * 1000:.0f} mm")
        print("\n开始检测角点...")

        valid_count = 0
        invalid_pairs = []

        for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
            # 读取图像
            img_left = cv2.imread(left_path)
            img_right = cv2.imread(right_path)

            if img_left is None or img_right is None:
                print(f"图像对 {i}: 读取失败")
                continue

            # 获取图像尺寸（只记录第一次）
            if self.image_size is None:
                self.image_size = img_left.shape[:2][::-1]  # (width, height)
                print(f"图像尺寸: {self.image_size[0]} x {self.image_size[1]} 像素")

            # 转换为灰度图
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            # 查找棋盘格角点
            ret_left, corners_left = cv2.findChessboardCorners(
                gray_left, self.chessboard_size, None
            )
            ret_right, corners_right = cv2.findChessboardCorners(
                gray_right, self.chessboard_size, None
            )

            if ret_left and ret_right:
                # 亚像素精确化
                corners_left = cv2.cornerSubPix(
                    gray_left, corners_left, (11, 11), (-1, -1), self.criteria
                )
                corners_right = cv2.cornerSubPix(
                    gray_right, corners_right, (11, 11), (-1, -1), self.criteria
                )

                # 存储点集
                self.objpoints.append(self.objp)
                self.imgpoints_left.append(corners_left)
                self.imgpoints_right.append(corners_right)

                valid_count += 1

                # 显示检测结果（可选）
                if i % 5 == 0:  # 每5对显示一次
                    self._show_corners(img_left, img_right, corners_left, corners_right, i)

                print(f"✓ 图像对 {i}: 角点检测成功 (有效: {valid_count})")
            else:
                invalid_pairs.append(i)
                print(f"✗ 图像对 {i}: 角点检测失败 (L:{ret_left}, R:{ret_right})")

        print(f"\n角点检测完成!")
        print(f"有效图像对: {valid_count}/{len(left_images)}")

        if invalid_pairs:
            print(f"无效图像对索引: {invalid_pairs[:10]}{'...' if len(invalid_pairs) > 10 else ''}")

        if valid_count < 10:
            print("\n⚠️ 警告: 有效图像对不足10对，建议至少采集15-20对以获得准确标定")
        elif valid_count >= 15:
            print("✓ 图像数量充足，可以进行标定")

        return valid_count

    def _show_corners(self, img_left, img_right, corners_left, corners_right, index):
        """显示角点检测结果"""
        # 绘制角点
        img_left_copy = img_left.copy()
        img_right_copy = img_right.copy()
        cv2.drawChessboardCorners(img_left_copy, self.chessboard_size, corners_left, True)
        cv2.drawChessboardCorners(img_right_copy, self.chessboard_size, corners_right, True)

        # 并排显示
        combined = np.hstack((img_left_copy, img_right_copy))

        # 添加文字信息
        cv2.putText(combined, f"Image Pair: {index}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, f"Valid Pairs: {len(self.objpoints)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, f"Chessboard: {self.chessboard_size[0]}x{self.chessboard_size[1]}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 调整显示大小
        h, w = combined.shape[:2]
        if w > 1920:
            scale = 1920 / w
            new_w = 1920
            new_h = int(h * scale)
            combined = cv2.resize(combined, (new_w, new_h))

        cv2.imshow('Chessboard Detection', combined)
        cv2.waitKey(500)  # 显示0.5秒

    def calibrate_single(self):
        """单目标定"""
        print("\n" + "=" * 60)
        print("开始单目标定...")
        print("=" * 60)

        # 左相机标定
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, self.image_size, None, None
        )

        # 右相机标定
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, self.image_size, None, None
        )

        print(f"\n左相机标定结果:")
        print(f"  重投影误差: {ret_left:.4f} 像素")
        print(f"  内参矩阵:\n{mtx_left}")
        print(f"  畸变系数: {dist_left.ravel()}")

        print(f"\n右相机标定结果:")
        print(f"  重投影误差: {ret_right:.4f} 像素")
        print(f"  内参矩阵:\n{mtx_right}")
        print(f"  畸变系数: {dist_right.ravel()}")

        return {
            'left': (mtx_left, dist_left),
            'right': (mtx_right, dist_right),
            'ret_left': ret_left,
            'ret_right': ret_right
        }

    def calibrate_stereo(self, mtx_left, dist_left, mtx_right, dist_right):
        """双目标定"""
        print("\n" + "=" * 60)
        print("开始双目标定...")
        print("=" * 60)

        # 双目立体标定
        flags = cv2.CALIB_FIX_INTRINSIC  # 固定内参，只优化外参
        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints,
            self.imgpoints_left,
            self.imgpoints_right,
            mtx_left, dist_left,
            mtx_right, dist_right,
            self.image_size,
            criteria=criteria_stereo,
            flags=flags
        )

        print(f"\n双目标定结果:")
        print(f"  重投影误差: {ret:.4f} 像素")
        print(f"  旋转矩阵 R:\n{R}")
        print(f"  平移向量 T:\n{T}")
        print(f"  基线距离: {np.linalg.norm(T):.3f} 米 ({np.linalg.norm(T) * 1000:.1f} mm)")
        print(f"  本质矩阵 E:\n{E}")
        print(f"  基础矩阵 F:\n{F}")

        # 立体校正
        print("\n开始立体校正...")
        R_left, R_right, P_left, P_right, Q, _, _ = cv2.stereoRectify(
            mtx_left, dist_left,
            mtx_right, dist_right,
            self.image_size,
            R, T,
            None, None, None, None, None,
            flags=cv2.CALIB_ZERO_DISPARITY
        )

        # 计算映射矩阵
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            mtx_left, dist_left, R_left, P_left, self.image_size, cv2.CV_32FC1
        )
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            mtx_right, dist_right, R_right, P_right, self.image_size, cv2.CV_32FC1
        )

        print("立体校正完成!")

        return {
            'ret': ret,
            'mtx_left': mtx_left,
            'dist_left': dist_left,
            'mtx_right': mtx_right,
            'dist_right': dist_right,
            'R': R,
            'T': T,
            'E': E,
            'F': F,
            'R_left': R_left,
            'R_right': R_right,
            'P_left': P_left,
            'P_right': P_right,
            'Q': Q,
            'map_left_x': map_left_x,
            'map_left_y': map_left_y,
            'map_right_x': map_right_x,
            'map_right_y': map_right_y
        }

    def save_calibration(self, stereo_params, filename='stereo_calibration.pkl'):
        """保存标定结果"""
        # 保存为 pickle 文件
        with open(filename, 'wb') as f:
            pickle.dump(stereo_params, f)
        print(f"\n标定结果已保存至: {filename}")

        # 同时保存为 npz 文件（方便其他程序读取）
        npz_filename = filename.replace('.pkl', '.npz')
        np.savez(npz_filename,
                 mtx_left=stereo_params['mtx_left'],
                 dist_left=stereo_params['dist_left'],
                 mtx_right=stereo_params['mtx_right'],
                 dist_right=stereo_params['dist_right'],
                 R=stereo_params['R'],
                 T=stereo_params['T'],
                 Q=stereo_params['Q'])
        print(f"标定结果已保存至: {npz_filename}")

    def load_calibration(self, filename='stereo_calibration.pkl'):
        """加载标定结果"""
        with open(filename, 'rb') as f:
            return pickle.load(f)


def verify_calibration(stereo_params, left_dir, right_dir, num_pairs=5):
    """验证标定结果：显示校正后的图像"""
    print("\n" + "=" * 60)
    print("验证标定结果...")
    print("=" * 60)

    # 获取映射矩阵
    map_left_x = stereo_params['map_left_x']
    map_left_y = stereo_params['map_left_y']
    map_right_x = stereo_params['map_right_x']
    map_right_y = stereo_params['map_right_y']

    # 获取图像列表
    left_images = sorted(glob.glob(os.path.join(left_dir, '*.png')))[:num_pairs]
    right_images = sorted(glob.glob(os.path.join(right_dir, '*.png')))[:num_pairs]

    print(f"显示 {len(left_images)} 对图像的校正结果...")
    print("观察绿色水平线是否对齐，如果对齐则标定成功")

    for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        if img_left is None or img_right is None:
            continue

        # 校正图像
        rect_left = cv2.remap(img_left, map_left_x, map_left_y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, map_right_x, map_right_y, cv2.INTER_LINEAR)

        # 并排显示，并绘制水平线检查对齐
        combined = np.hstack((rect_left, rect_right))
        h, w = combined.shape[:2]

        # 绘制水平线（每50像素一条）
        for y in range(0, h, 50):
            cv2.line(combined, (0, y), (w, y), (0, 255, 0), 1)

        # 添加文字信息
        cv2.putText(combined, f"Rectified Image Pair {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Green lines should align horizontally", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # 调整显示大小
        if w > 1920:
            scale = 1920 / w
            new_w = 1920
            new_h = int(h * scale)
            combined = cv2.resize(combined, (new_w, new_h))

        cv2.imshow('Rectified Stereo Pair - Press any key for next', combined)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("验证完成!")


def main():
    """主函数"""
    print("=" * 60)
    print("ZED X 双目相机标定程序")
    print("=" * 60)

    # ========== 配置参数（根据你的棋盘格修改） ==========
    # 棋盘格参数
    chessboard_size = (11, 8)  # 内角点数 (宽度方向11个, 高度方向8个)
    square_size = 0.04  # 方格边长（米），40mm = 0.04米

    # 图像路径（根据你的实际保存路径）
    left_dir = "zed_photos/left"  # 左相机图像目录
    right_dir = "zed_photos/right"  # 右相机图像目录

    # 打印配置信息
    print(f"\n标定配置:")
    print(f"  棋盘格内角点: {chessboard_size[0]} x {chessboard_size[1]}")
    print(
        f"  棋盘格实际尺寸: {chessboard_size[0] * square_size * 1000:.0f} x {chessboard_size[1] * square_size * 1000:.0f} mm")
    print(f"  方格边长: {square_size * 1000:.0f} mm")
    print(f"  左图像目录: {left_dir}")
    print(f"  右图像目录: {right_dir}")

    # 检查目录是否存在
    if not os.path.exists(left_dir):
        print(f"\n❌ 错误: 左图像目录不存在: {left_dir}")
        return
    if not os.path.exists(right_dir):
        print(f"\n❌ 错误: 右图像目录不存在: {right_dir}")
        return

    # ========== 开始标定 ==========
    # 1. 创建标定器
    calibrator = ZedXStereoCalibration(chessboard_size, square_size)

    # 2. 加载图像并检测角点
    valid_pairs = calibrator.load_images(left_dir, right_dir)

    if valid_pairs < 10:
        print("\n❌ 有效图像对不足10对，无法进行标定")
        print("请确保:")
        print("  1. 图像保存在正确的目录中")
        print("  2. 棋盘格角点能够被清晰检测")
        print("  3. 至少采集15-20对不同姿态的标定板图像")
        return

    # 3. 单目标定
    single_params = calibrator.calibrate_single()

    # 4. 双目标定
    stereo_params = calibrator.calibrate_stereo(
        single_params['left'][0], single_params['left'][1],
        single_params['right'][0], single_params['right'][1]
    )

    # 5. 保存标定结果
    calibrator.save_calibration(stereo_params)

    # 6. 验证标定结果
    verify_calibration(stereo_params, left_dir, right_dir)

    print("\n" + "=" * 60)
    print("标定完成!")
    print("=" * 60)

    # 输出标定参数总结
    print("\n标定参数总结:")
    print(f"  重投影误差: {stereo_params['ret']:.4f} 像素")
    print(
        f"  基线距离: {np.linalg.norm(stereo_params['T']):.3f} 米 ({np.linalg.norm(stereo_params['T']) * 1000:.1f} mm)")
    print(f"  左相机焦距: fx={stereo_params['mtx_left'][0, 0]:.2f}, fy={stereo_params['mtx_left'][1, 1]:.2f} 像素")
    print(f"  右相机焦距: fx={stereo_params['mtx_right'][0, 0]:.2f}, fy={stereo_params['mtx_right'][1, 1]:.2f} 像素")
    print(f"  主点坐标: cx={stereo_params['mtx_left'][0, 2]:.1f}, cy={stereo_params['mtx_left'][1, 2]:.1f} 像素")
    print(f"\n标定文件已保存:")
    print(f"  - stereo_calibration.pkl (Python pickle格式)")
    print(f"  - stereo_calibration.npz (NumPy格式)")


if __name__ == "__main__":
    main()
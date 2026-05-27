import cv2
import os
import glob
import time  # 1. 引入 time 模块


def bicubic_upsample_folder_with_timer(input_folder, output_folder, scale_factor=4):
    """
    读取输入文件夹中的图片，使用 Bicubic 插值上采样，并计算平均重建时间。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_paths:
        print(f"警告: 在 {input_folder} 中未找到支持的图像文件。")
        return

    print(f"开始处理，共发现 {len(image_paths)} 张图片...")

    # --- 统计变量 ---
    total_reconstruction_time = 0.0
    processed_count = 0

    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            new_dim = (w * scale_factor, h * scale_factor)

            # --- 计时开始 (仅计算核心算法时间) ---
            start_time = time.perf_counter()

            # Bicubic 上采样
            upsampled_img = cv2.resize(img, new_dim, interpolation=cv2.INTER_CUBIC)

            # --- 计时结束 ---
            end_time = time.perf_counter()

            # 累加单张耗时
            current_time_cost = end_time - start_time
            total_reconstruction_time += current_time_cost
            processed_count += 1

            # 保存图片 (I/O 操作不计入重建时间)
            base_name = os.path.basename(img_path)
            filename, ext = os.path.splitext(base_name)
            new_filename = f"{filename}_x{scale_factor}_bicubic{ext}"
            save_path = os.path.join(output_folder, base_name)
            cv2.imwrite(save_path, upsampled_img)

            print(f"[{processed_count}] {new_filename} -> {new_dim} | 耗时: {current_time_cost * 1000:.2f} ms")

        except Exception as e:
            print(f"处理图片 {img_path} 时发生错误: {e}")

    # --- 计算并输出平均时间 ---
    print("-" * 30)
    if processed_count > 0:
        avg_time = total_reconstruction_time / processed_count
        print(f"成功处理图片: {processed_count} 张")
        print(f"总算法耗时: {total_reconstruction_time:.4f} s")
        print(f"平均每张重建用时: {avg_time * 1000:.4f} ms (毫秒)")
    else:
        print("未成功处理任何图片。")
    print("-" * 30)


# --- 使用示例 ---
if __name__ == "__main__":
    # 请在这里修改你的实际路径
    input_dir = '/media/dell/T7 Shield/wsisr_pipeline/dataset/multi_mag/external_cptac/lr_X8'  # 输入文件夹
    output_dir = '/media/dell/data/zhangv1/WSISR/method_comparison/patch-multimag/bicubic/cptac/8'  # 输出文件夹

    # # 确保有一个示例输入文件夹以免报错 (仅用于演示)
    # if not os.path.exists(input_dir):
    #     os.makedirs(input_dir)
    #     print(f"提示: 请将测试图片放入 {input_dir} 文件夹中")

    # 调用函数
    bicubic_upsample_folder_with_timer(input_dir, output_dir, scale_factor=8)

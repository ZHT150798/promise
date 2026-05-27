import os
import re
import openslide
import shutil

def approximate_magnification(mpp_x):
    if mpp_x is not None:
        if mpp_x < 0.16:
            return 80
        elif mpp_x < 0.2:
            return 60
        elif mpp_x < 0.3:
            return 40
        elif mpp_x < 0.6:
            return 20
        elif mpp_x < 1.2:
            return 10
        elif mpp_x < 2.4:
            return 5
        else:
            raise ValueError(f"Identified mpp is very low: mpp={mpp_x}. Most WSIs are at 20x, 40x magnfication.")

def parse_magnification(mag_value):
    """将不同格式的放大倍数转换为浮点数"""
    if isinstance(mag_value, (int, float)):
        return float(mag_value)

    try:
        # 尝试直接转换数字
        return float(mag_value)
    except ValueError:
        # 使用正则表达式提取数字部分（处理类似"40x"的情况）
        match = re.search(r'(\d+\.?\d*)', str(mag_value))
        return float(match.group(1)) if match else None


def get_magnification(wsi_path, custom_mpp_keys=None):
    """获取单个WSI的放大倍数，支持多种元数据格式和解析方法"""
    try:
        slide = openslide.OpenSlide(wsi_path)
        props = slide.properties

        # # Step 1: 先尝试常见放大倍数属性键
        # mag_keys = [
        #     'aperio.AppMag',  # Aperio
        #     'openslide.objective-power',  # OpenSlide标准
        #     'hamamatsu.SourceLens',  # Hamamatsu
        #     'tiff.ImageDescription',  # Olympus或其他厂商
        # ]
        #
        # for key in mag_keys:
        #     if key in props:
        #         return parse_magnification(props[key])  # 可根据字符串处理规则自定义

        # Step 2: 如果没有放大倍数，尝试通过MPP推算
        mpp_keys = [
            openslide.PROPERTY_NAME_MPP_X,  # 'openslide.mpp-x'
            'openslide.mirax.MPP',
            'aperio.MPP',
            'hamamatsu.XResolution',
            'openslide.comment',
            'openslide.mpp-x'
            ,
        ]

        if custom_mpp_keys:
            mpp_keys.extend(custom_mpp_keys)

        for key in mpp_keys:
            if key in props:
                try:
                    mpp = float(props[key])
                    return approximate_magnification(mpp)
                except ValueError:
                    continue

        # Step 3: 从 TIFF 头部信息中推断 MPP
        x_resolution = props.get('tiff.XResolution')
        unit = props.get('tiff.ResolutionUnit')
        if x_resolution and unit:
            try:
                if unit.lower() == 'centimeter':
                    mpp = 10000 / float(x_resolution)
                    return approximate_magnification(mpp)
                elif unit.upper() == 'INCH':
                    mpp = 25400 / float(x_resolution)
                    return approximate_magnification(mpp)
            except ValueError:
                pass

        # Step 4: 所有方法失败
        print(f"Warning: 无法从 {wsi_path} 中提取放大倍数")
        return None

    except Exception as e:
        print(f"Error processing {wsi_path}: {str(e)}")
        return None

    finally:
        if 'slide' in locals():
            slide.close()


def check_wsi_magnification(folder_path, target_mag, move=False):
    supported_ext = ['.svs', '.tiff', '.tif', '.ndpi', '.mrxs', '.scn']
    target_mag = float(target_mag)

    mismatch_count = 0
    mismatch_files = []
    total_checked = 0

    for filename in os.listdir(folder_path):
        if not any(filename.lower().endswith(ext) for ext in supported_ext):
            continue

        wsi_path = os.path.join(folder_path, filename)
        mag = get_magnification(wsi_path)
        total_checked += 1

        if mag is None:
            print(f"  Warning: Could not determine magnification for {filename}")
            mismatch_count += 1
            mismatch_files.append((filename, "Unknown"))
        elif abs(mag - target_mag) > 0.1:
            mismatch_count += 1
            mismatch_files.append((filename, mag))

            # ====== 这里是“移动”逻辑 ======
            if move and isinstance(mag, (int, float)):
                subdir_name = f"{int(mag)}x"
                dest_dir = os.path.join(folder_path, subdir_name)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, filename)
                try:
                    shutil.move(wsi_path, dest_path)
                    print(f"Moved: {filename} -> {dest_dir}")
                except Exception as e:
                    print(f"  Error moving {filename}: {e}")
        else:
            pass  # 匹配就不管了

    print("\n===== Final Report =====")
    print(f"Total WSI files checked: {total_checked}")
    print(f"Files with mismatch magnification: {mismatch_count}")
    print("Mismatch files list:")
    for f, mag in mismatch_files:
        print(f"  {f} -> Magnification: {mag}")


if __name__ == "__main__":
    folder_path = "/media/dell/data/zhangv1/WSISR/WSI_SR_Protocol/pathologist_assess/breast/BJSZHP/slide"  # 修改为你的路径
    target_magnification = 40 # 目标放大倍数
    check_wsi_magnification(folder_path, target_magnification, move=False)

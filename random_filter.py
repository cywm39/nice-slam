import os
import random
import cv2
import numpy as np

def modify_depth_images(input_folder, output_folder, invalid_value):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的深度图文件列表
    file_list = os.listdir(input_folder)

    for file_name in file_list:
        # 构建完整的文件路径
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # 读取深度图像
        depth_image = cv2.imread(input_path, cv2.IMREAD_ANYDEPTH)

        # 获取图像尺寸
        height, width = depth_image.shape

        # 计算总像素数量
        num_pixels = height * width

        # 计算要修改的像素数量
        num_modified_pixels = int(num_pixels * 0.2)

        # 创建一个包含所有像素索引的列表
        all_pixels = np.arange(num_pixels)

        # 随机选择要修改的像素索引，确保不重复
        modified_pixels = random.sample(all_pixels.tolist(), num_modified_pixels)

        # 将修改的像素索引转换为对应的行和列坐标
        modified_rows = modified_pixels // width
        modified_cols = modified_pixels % width

        # 在图像中将选中的像素的深度值设为无效值
        depth_image[modified_rows, modified_cols] = invalid_value

        # 保存修改后的深度图像
        cv2.imwrite(output_path, depth_image)

# 指定输入文件夹和输出文件夹路径
input_folder = './Datasets/Apartment/depth'
output_folder = './Datasets/Apartment/random_filter_20'

# 调用函数进行深度图像修改和保存
modify_depth_images(input_folder, output_folder, invalid_value=0)

# import numpy as np
# import cv2

# # 读取深度图像
# depth_image = cv2.imread('./Datasets/Apartment/depth/00000.png', cv2.IMREAD_ANYDEPTH)

# # 获取深度图像的行数和列数
# rows, cols = depth_image.shape

# # 将深度值输出到 txt 文件
# with open('./depth_values.txt', 'w') as file:
#     for row in range(rows):
#         for col in range(cols):
#             depth_value = depth_image[row, col]
#             file.write(f'{depth_value}\t')
#         file.write('\n')

# from src import config
# cfg = config.load_config(
#         'configs/Apartment/apartment.yaml', 'configs/nice_slam.yaml')
# print(cfg['cam']['H'])
# print(cfg['mapping']['lr_first_factor'])
# print(cfg['cam']['H'])
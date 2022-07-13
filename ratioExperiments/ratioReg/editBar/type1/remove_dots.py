'''
@Author: WANG Maonan
@Date: 2021-06-17 01:38:57
@Description: 将 dots 删除
@LastEditTime: 2021-08-16 12:35:15
'''
import copy
import numpy as np
import logging

def remove_black_dot(im, dots_positions):
    """将 black-dot 删除
    
    Args:
        im (array): bar chart 数据
        dots_positions (list): 图中两个 black-dots 的位置, 例如 [[96, 14], [96, 23]], 分别是 y 轴和 x 轴
    """
    image = copy.deepcopy(im)

    # 清除两个 bar 的 black-dot
    for dots_position in dots_positions:
        image[dots_position[0], dots_position[1]] = 0 # 将 black-dots 的位置修改为 0

    return image


def change_numpy2removedots(
                        image_numpy_file:str, 
                        label_numpy_file:str,
                        bar_height_numpy_file:str,
                        dots_positions_numpy_file:str
                        ):
    """将原始的 image 转换为 移动点的 image

    Args:
        image_numpy_file (str): 存储原始 image 的 npy 文件的路径
        label_numpy_file (str): 存储原始 label 的 npy 文件
        bar_height_numpy_file (str): 存储 bar height 的 npy 文件
        dots_positions_numpy_file (str): 存储原始 image 中的 dots locations 的 npy 文件的路径
    """
    logger = logging.getLogger(__name__)

    image_numpy_data = np.load(image_numpy_file) # 获得 image numpy 文件
    label_numpy_data = np.load(label_numpy_file)
    bar_height_numpy_data = np.load(bar_height_numpy_file)
    dots_positions_numpy_data = np.load(dots_positions_numpy_file) # dot position 的 numpy 文件

    for image_index, image in enumerate(image_numpy_data):
        new_image = remove_black_dot(image[0], dots_positions_numpy_data[image_index]) # 修改原始的 image 图像
        image_numpy_data[image_index][0] = new_image # 替换原始数据

    logger.info('将原始数据集修改为 remove-dots.')
    return (image_numpy_data, label_numpy_data, bar_height_numpy_data, dots_positions_numpy_data)
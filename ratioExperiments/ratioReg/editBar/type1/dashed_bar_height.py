'''
@Author: WANG Maonan
@Date: 2021-06-17 03:24:33
@Description: 将垂直的竖线变为虚线
@LastEditTime: 2021-08-16 12:38:13
'''
import copy
import numpy as np
import logging

def dashed_bar_height(im, height_list, dots_positions):
    """将垂直线变为虚线
    Args:
        im (array): bar chart 数据
        height_list (array): bar chart 的每一个 bar height, 需要用来推算横轴的高度
        dots_positions (list): 图中两个 black-dots 的位置, 例如 [[96, 14], [96, 23]]
    """
    image = copy.deepcopy(im)

    dots_position2index = {5:0, 14:1, 23:2, 32:3, 41:4, 58:5, 67:6, 76:7, 85:8, 94:9} # 从 dots-position 定位到 bar index
    line_index2position = {
                            0:(2,9), 1:(11,18), 2:(20,27), 3:(29,36), 4:(38,45), 
                            5:(55,62), 6:(64,71), 7:(73,80), 8:(82,89), 9:(91,98)
                           } # 根据 bar index 得到 vertical line 的坐标
                           
    # 确定 black-dots 是在哪两个 bar 上面
    left_dot_index, right_dot_index = [dots_position2index[dots_position[1]] for dots_position in dots_positions] # 首先获得当前的 black-dots 在哪个 bar 上面

    # 获得 black-dots 所在的 bar height
    left_bar_height = int(height_list[left_dot_index]) # 获得 left black-dots 对应的 bar 的高度
    right_bar_height = int(height_list[right_dot_index]) # 获得 right black-dots 对应的 bar 的高度

    # 获得 black-dots 所在的 bar 的垂直线的坐标
    left_bar_positions = line_index2position[left_dot_index]
    right_bar_positions = line_index2position[right_dot_index]
     
    # 移除 bar line
    for dot_index, bar_height in zip([left_dot_index, right_dot_index], [left_bar_height, right_bar_height]): # 依次获得 left-dot-index 和 right-dot-index
        (left_bar_position, right_bar_position) = line_index2position[dot_index]
        for _i in range(1, bar_height, 2):
            image[100-_i, left_bar_position] = 0
            image[100-_i, right_bar_position] = 0
    
    # 增加 horizontal line
    image[100-left_bar_height, left_bar_positions[0]:left_bar_positions[1]+1] = 1 # 去除第一个水平线, 需要注意去除 top-bar 不能改变高度
    image[100-right_bar_height, right_bar_positions[0]:right_bar_positions[1]+1] = 1 # 去除第二个水平线    

    return image


def change_numpy2dashedBarHeight(
                    image_numpy_file:str,
                    label_numpy_file:str,
                    bar_height_numpy_file:str, 
                    dots_positions_numpy_file:str):
    """将原始的 image 转换为 移动点的 image
    """
    logger = logging.getLogger(__name__)

    image_numpy_data = np.load(image_numpy_file) # 获得 bar image 的 numpy 文件
    label_numpy_data = np.load(label_numpy_file) # 获得 label 的 numpy 文件
    bar_height_data = np.load(bar_height_numpy_file) # 获得 bar height 的 numpy 文件
    dots_positions_numpy_data = np.load(dots_positions_numpy_file) # dot position 的 numpy 文件

    for image_index, image in enumerate(image_numpy_data):
        new_image = dashed_bar_height(im = image[0], 
                                         height_list=bar_height_data[image_index],
                                         dots_positions = dots_positions_numpy_data[image_index]
                                        )

        image_numpy_data[image_index][0] = new_image # 修改原始的 image 图像

    logger.info('将原始数据集修改为 dashed bar-height.')
    return (image_numpy_data, label_numpy_data, bar_height_data, dots_positions_numpy_data)
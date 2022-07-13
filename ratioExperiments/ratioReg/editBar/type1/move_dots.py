'''
@Author: WANG Maonan
@Date: 2021-06-15 21:04:44
@Description: 修改原始数据, 得到移动 dot 之后的数据
@LastEditTime: 2021-08-16 12:37:05
'''
import copy
import numpy as np
np.random.seed(777)
import logging

from ...ClevelandMcGill.bar_figure_type1 import BarFigure_type1

def move_black_dot(im, height_list, dots_positions, position_left, position_right):
    """将 black-dot 移动到指定的 index
    
    Args:
        im (array): bar chart 数据
        height_list (array): bar chart 的每一个 bar height
        dots_positions (array): 两个 dots 的 y,x 坐标, 例如 [[96, 15], [96, 25]]
        position_left (int): 需要移动到「左侧的 bar 的位置」, 即新的左侧 bar 的 index
        position_right (int): 需要移动到「右侧的 bar 的位置」
    """
    image = copy.deepcopy(im)
    dots_index2position = {0:5, 1:14, 2:23, 3:32, 4:41, 5:58, 6:67, 7:76, 8:85, 9:94} # 每一个 bar 中 dots 对应的坐标
    dots_position2index = {5:0, 14:1, 23:2, 32:3, 41:4, 58:5, 67:6, 76:7, 85:8, 94:9}
    
    # 首先确定 dots 在哪个 bar 上面
    left_dot_position, right_dot_position = [dots_position2index[dots_position[1]] for dots_position in dots_positions] # 首先获得当前的 black-dots 在哪个 bar 上面
    left_dot_height, right_dot_height = [dots_position[0] for dots_position in dots_positions] # 获得当前的 black-dots 的 height

    left_bar_height = height_list[left_dot_position] # 当前左侧 black-dot 的 bar height
    right_bar_height = height_list[right_dot_position] # 当前右侧 black-dot 的 bar height
    
    # 交换两个 bar 的 height, 更新 height list
    height_list[left_dot_position] = height_list[position_left]
    height_list[right_dot_position] = height_list[position_right]

    height_list[position_left] = left_bar_height
    height_list[position_right] = right_bar_height

    image, all_values, dots_positions = BarFigure_type1.data_to_custom_bar(
                                            height_data=height_list, 
                                            position = ((position_left, left_dot_height), (position_right, right_dot_height))
                                        )
    assert all(height_list == all_values), 'bar height 计算错误.'

    new_height_list = [height_list[left_dot_position], height_list[right_dot_position]]
    new_ratio = min(new_height_list)/max(new_height_list) # 计算新的 ratio

    return image, new_ratio, height_list, dots_positions


def change_numpy2movedots(
                    image_numpy_file:str,
                    label_numpy_file:str,
                    bar_height_numpy_file:str, 
                    dots_positions_numpy_file:str):
    """将原始的 image 转换为 移动点的 image
    """
    logger = logging.getLogger(__name__)

    image_numpy_data = np.load(image_numpy_file) # 获得 image 的 numpy 文件
    label_numpy_data = np.load(label_numpy_file) # 获得 label 的 numpy 文件
    bar_height_data = np.load(bar_height_numpy_file) # 获得 bar height 的 numpy 文件
    dots_positions_numpy_data = np.load(dots_positions_numpy_file) # dot position 的 numpy 文件
    
    for image_index, image in enumerate(image_numpy_data):
        bar_indexs = np.random.choice(list(range(10)), 2, replace=False) # 随机两个 bar index
        new_image, new_ratio, new_height, new_dots_positions = move_black_dot(
                                                                        im = image[0], # 这里原始图像大小为 1*100*100, 需要变为 100*100
                                                                        height_list = bar_height_data[image_index], 
                                                                        dots_positions = dots_positions_numpy_data[image_index],
                                                                        position_left = bar_indexs[0], 
                                                                        position_right = bar_indexs[1]
                                                                    ) 

        image_numpy_data[image_index][0] = new_image # 修改原始的 image 图像
        # label_numpy_data[image_index][0] = new_ratio # 修改 label
        bar_height_data[image_index] = new_height # 修改 bar height
        dots_positions_numpy_data[image_index] = np.array(new_dots_positions) # 修改 dots postion

    logger.info('将原始数据集修改为 move-dots.')
    return (image_numpy_data, label_numpy_data, bar_height_data, dots_positions_numpy_data)
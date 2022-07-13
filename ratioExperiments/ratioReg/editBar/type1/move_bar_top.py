'''
Author: WANG Maonan
Date: 2021-07-25 02:00:43
Description: 移动 bar-top 的位置, 将 bar-top 向下移动
@LastEditTime: 2021-08-16 12:38:32
'''
import copy
import torch
import numpy as np
np.random.seed(777)
import logging

def move_barTop(im, height_list, dots_positions, height_minus=None, bar_index=None):
    """将 black-dots 对应的 bar 的 bar-top 移动位置, 使其和 bar height 有冲突

    Args:
        im (array): bar chart 数据
        height_list (array): bar chart 的每一个 bar height, 需要用来推算横轴的高度
        dots_positions (list): 图中两个 black-dots 的位置, 例如 [[96, 14], [96, 23]]
        height_minus (int, optional): bar-top 向下移动的距离. Defaults to None.
        bar_index (int, optional): 修改哪一个 bar 的 bar-top. Defaults to None.

    Returns:
        im: 修改后的图像
    """
    image = copy.deepcopy(im)
    _height_list = copy.deepcopy(height_list)
    
    dots_position2index = {5:0, 14:1, 23:2, 32:3, 41:4, 58:5, 67:6, 76:7, 85:8, 94:9} # 从 dots-position 定位到 bar index
    line_index2position = {
                            0:(2,9), 1:(11,18), 2:(20,27), 3:(29,36), 4:(38,45), 
                            5:(55,62), 6:(64,71), 7:(73,80), 8:(82,89), 9:(91,98)
                           } # 根据 bar index 得到 vertical line 的坐标
                           
    # 确定 black-dots 是在哪两个 bar 上面
    left_dot_index, right_dot_index = [dots_position2index[dots_position[1]] for dots_position in dots_positions] # 首先获得当前的 black-dots 在哪个 bar 上面

    # 随机选择一个 bar
    if bar_index == None: # 是否指定了 select bar index
        selected_bar_index = np.random.choice([left_dot_index, right_dot_index], 1)[0] # 被选中进行减少高度的 bar
    else:
        selected_bar_index = bar_index
    
    # 确定 bar-top 移动的距离
    if height_minus == None: # 是否指定了减少的长度
        if _height_list[selected_bar_index]>=6: # 如果 bar height 太短, 则不需要再进行裁剪
            minus = np.random.choice(list(range(0, int(_height_list[selected_bar_index]-5)))) # 随机需要移动的长度
        else:
            minus = 0
    else:
        minus = height_minus

    # 获得 selected bar 所在的 bar height 和 positions
    selected_bar_height = int(_height_list[selected_bar_index]) # 获得 selected bar 对应的 bar 的高度
    selected_bar_positions = line_index2position[selected_bar_index] # 获得 black-dots 所在的 bar 的垂直线的坐标

    image[100-selected_bar_height+minus, selected_bar_positions[0]+1:selected_bar_positions[1]] = 1 # 增加 bar-top
    if minus != 0: # 将原来的 bar-top 去除, 需要注意 bar-top 移动不能为 0
        image[100-selected_bar_height, selected_bar_positions[0]+1:selected_bar_positions[1]] = 0 # 去除水平线, 需要注意去除 top-bar 不能改变高度

    # 按照 bar-top 来计算 ratio
    _height_list[selected_bar_index] -= minus
    dots_height = (_height_list[left_dot_index], _height_list[right_dot_index]) # 新的两个 bar height

    return image, min(dots_height)/max(dots_height)


def change_numpy2moveBarTop(                    
                    image_numpy_file:str,
                    label_numpy_file:str,
                    bar_height_numpy_file:str, 
                    dots_positions_numpy_file:str):
    """将原始的 image 转换为改变 bar-top 的 image
    """
    logger = logging.getLogger(__name__)

    image_numpy_data = np.load(image_numpy_file) # 获得 bar image 的 numpy 文件
    label_numpy_data = np.load(label_numpy_file) # 获得 label 的 numpy 文件
    bar_height_data = np.load(bar_height_numpy_file) # 获得 bar height 的 numpy 文件
    dots_positions_numpy_data = np.load(dots_positions_numpy_file) # dot position 的 numpy 文件

    for image_index, image in enumerate(image_numpy_data):
        new_image, new_ratio = move_barTop(
                                            im = image[0], 
                                            height_list = bar_height_data[image_index],
                                            dots_positions=dots_positions_numpy_data[image_index]
                                        )
        image_numpy_data[image_index][0] = new_image # 修改原始的 image 图像
        # label_numpy_data[image_index][0] = new_ratio # 修改 label


    logger.info('将原始数据集修改为 move bar-top.')
    return (image_numpy_data, label_numpy_data, bar_height_data, dots_positions_numpy_data)
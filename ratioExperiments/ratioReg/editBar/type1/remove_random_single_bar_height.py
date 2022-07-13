'''
@Author: WANG Maonan
@Date: 2021-08-30 05:01:37
@Description: 一共有 4 个 vertical line, 随机删除一个
@LastEditTime: 2021-08-30 09:59:05
'''
import copy
import numpy as np
np.random.seed(777)
import logging

def remove_random_single_bar_height(im, height_list, dots_positions, vertical_index=None):
    """随机删除一个竖着的 bar line
    
    Args:
        im (array): bar chart 数据
        height_list (array): bar chart 的每一个 bar height
    """
    image = copy.deepcopy(im)
    dots_position2index = {5:0, 14:1, 23:2, 32:3, 41:4, 58:5, 67:6, 76:7, 85:8, 94:9}
    line_index2position = {
                        0:(2,9), 1:(11,18), 2:(20,27), 3:(29,36), 4:(38,45), 
                        5:(55,62), 6:(64,71), 7:(73,80), 8:(82,89), 9:(91,98)
                        } # 根据 bar index 得到 vertical line 的坐标
    
    # 首先确定 dots 在哪个 bar 上面
    left_dot_position, right_dot_position = [dots_position2index[dots_position[1]] for dots_position in dots_positions] # 首先获得当前的 black-dots 在哪个 bar 上面

    left_bar_height = int(height_list[left_dot_position]) # 当前左侧 black-dot 的 bar height
    right_bar_height = int(height_list[right_dot_position]) # 当前右侧 black-dot 的 bar height
    
    # 去除 dots
    # for dots_position in dots_positions:
    #     image[dots_position[0], dots_position[1]] = 0 # 将 black-dots 的位置修改为 0
        
    # 去除水平的 bar
    left_bar_positions = line_index2position[left_dot_position] # 获得水平线的坐标, 例如 0-->(2,9)
    right_bar_positions = line_index2position[right_dot_position]
    
    image[100-left_bar_height, left_bar_positions[0]+1:left_bar_positions[1]] = 0 # 去除第一个水平线, 需要注意去除 top-bar 不能改变高度
    image[100-right_bar_height, right_bar_positions[0]+1:right_bar_positions[1]] = 0 # 去除第二个水平线

    # 随机去掉一个 vertical line
    if vertical_index == None:
        vertical_index = np.random.choice(list(range(4)), 1, replace=False) # 要删除的一个 index
    if vertical_index[0] == 0:
        image[:, left_bar_positions[0]] = 0 # 左侧 bar 的 left line
    elif vertical_index[0] == 1:
        image[:, left_bar_positions[1]] = 0
    elif vertical_index[0] == 2:
        image[:, right_bar_positions[0]] = 0
    elif vertical_index[0] == 3:
        image[:, right_bar_positions[1]] = 0

    return image


def change_numpy2removeRandomSingleBarHeight(
                                image_numpy_file:str,
                                label_numpy_file:str,
                                bar_height_numpy_file:str, 
                                dots_positions_numpy_file:str
                            ):
    """将原始的 image 转换为「只保留左侧的 bar」
    """
    logger = logging.getLogger(__name__)

    image_numpy_data = np.load(image_numpy_file) # 获得 bar image 的 numpy 文件
    label_numpy_data = np.load(label_numpy_file) # 获得 label 的 numpy 文件
    bar_height_data = np.load(bar_height_numpy_file) # 获得 bar height 的 numpy 文件
    dots_positions_numpy_data = np.load(dots_positions_numpy_file) # dot position 的 numpy 文件
    

    for image_index, image in enumerate(image_numpy_data):
        new_image = remove_random_single_bar_height(im = image[0], 
                                            height_list = bar_height_data[image_index], 
                                            dots_positions = dots_positions_numpy_data[image_index],
                                        ) # 这里原始图像大小为 1*100*100, 需要变为 100*100

        image_numpy_data[image_index][0] = new_image # 修改原始的 image 图像


    logger.info('将原始数据集修改为 Remove Random Single Bar Heights.')
    return (image_numpy_data, label_numpy_data, bar_height_data, dots_positions_numpy_data)
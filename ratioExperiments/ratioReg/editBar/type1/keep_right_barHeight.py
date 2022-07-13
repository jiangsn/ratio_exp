'''
@Author: WANG Maonan
@Date: 2021-08-16 13:11:32
@Description: 修改 bar, 只保留右侧的一个 bar-height
@LastEditTime: 2021-08-16 13:47:18
'''
import sys
sys.path.append('/data/maonanwang/')

import copy
import os
import numpy as np
np.random.seed(777)
import torch

from ratioReg.TrafficLog.setLog import logger
from ratioReg.utils.save_image import save_ratio_image
from ratioReg.utils.calcBarHeight_Type1 import calc_type1_bar_heights
from ratioReg.utils.compareBarHeight import compare_bar_height

def keep_right_bar_height(im, height_list, dots_positions):
    """只保留右侧的竖着的 bar, 其余的都删除
    
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
    for dots_position in dots_positions:
        image[dots_position[0], dots_position[1]] = 0 # 将 black-dots 的位置修改为 0
        
    # 去除水平的 bar
    left_bar_positions = line_index2position[left_dot_position] # 获得水平线的坐标
    right_bar_positions = line_index2position[right_dot_position]
    
    image[100-left_bar_height, left_bar_positions[0]+1:left_bar_positions[1]] = 0 # 去除第一个水平线, 需要注意去除 top-bar 不能改变高度
    image[100-right_bar_height, right_bar_positions[0]+1:right_bar_positions[1]] = 0 # 去除第二个水平线

    # 去除 left bar
    image[:, left_bar_positions[0]] = 0 # 左侧 bar 的 left-bar-height
    image[:, right_bar_positions[0]] = 0 # 右侧 bar 的 left-bar-height

    return image


def change_numpy2keeprightbarHeight(image_numpy_file, height_numpy_file, dots_positions_numpy_file):
    """将原始的 image 转换为「只保留右侧的 bar」
    """
    image_numpy_data = np.load(image_numpy_file) # 获得 image 的 numpy 文件
    bar_height_data = np.load(height_numpy_file) # 获得 bar height 的 numpy 文件
    dots_positions_numpy_data = np.load(dots_positions_numpy_file) # dot position 的 numpy 文件
    

    for image_index, image in enumerate(image_numpy_data):
        new_image = keep_right_bar_height(im = image[0], 
                                            height_list = bar_height_data[image_index], 
                                            dots_positions = dots_positions_numpy_data[image_index],
                                        ) # 这里原始图像大小为 1*100*100, 需要变为 100*100

        image_numpy_data[image_index][0] = new_image # 修改原始的 image 图像

    image_tensor = torch.from_numpy(image_numpy_data)

    logger.info('将原始数据集修改为 move-dots.')
    return image_tensor

if __name__ == "__main__":
    image_numpy_file = './ratioRegression/exp_output/dataset/type1/fixed_bottom/0/val-image.npy'
    height_numpy_file = './ratioRegression/exp_output/dataset/type1/fixed_bottom/0/val-bar_height.npy'
    bar_heights = np.load(height_numpy_file)
    dots_positions_numpy_file = './ratioRegression/exp_output/dataset/type1/fixed_bottom/0/val-dots_positions.npy'
    image_tensor = change_numpy2keeprightbarHeight(image_numpy_file, height_numpy_file, dots_positions_numpy_file)

    # 将 tensor 转换为 np.array
    save_path = './ratioRegression/editBar/example/keep_right_barHeight'
    os.makedirs(save_path, exist_ok=True)
    image_indexs = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 10000]
    for image_index in image_indexs:
        convert_image = image_tensor.numpy()[image_index][0] # 修改后的图片
        raw_image = np.load(image_numpy_file)[image_index][0] # 原始的图片

        # 比较 bar height 是否相等
        converted_bar_height = calc_type1_bar_heights(convert_image) # 修改后的 bar height
        compare_bar_height(converted_bar_height, bar_heights[image_index]) # 比较一下两个 bar 的长度
        
        # 保存图像
        save_ratio_image(convert_image, os.path.join(save_path, '{}_convert_image.jpg'.format(image_index)))
        save_ratio_image(raw_image, os.path.join(save_path, '{}_raw_image.jpg'.format(image_index)))
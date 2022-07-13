'''
@Author: WANG Maonan
@Date: 2021-06-17 17:16:03
@Description: 增长较短的 bar, 但是保留原始的 top-bar
@LastEditTime: 2021-07-20 04:59:53
'''
import sys
sys.path.append('/data/maonanwang/')

import os
import copy
import torch
import numpy as np
np.random.seed(777)
import logging

from ratioReg.utils.save_image import save_ratio_image

def add_vertiacl_line_withTopbar(im, height_list, dots_positions, height_minus=None, bar_index=None):
    """增长较短的 bar, 同时把最短的 bar 的 bar-top 删除
    Args:
        im (array): bar chart 数据
        height_list (array): bar chart 的每一个 bar height
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

    # 随机选择一个 bar, 计算增加多少高度
    if bar_index == None:
        selected_bar_index = np.random.choice([left_dot_index, right_dot_index], 1)[0] # 被选中进行增加高度的 bar
    else:
        selected_bar_index = bar_index
    if height_minus == None:
        minus = int(np.abs(86-height_list[selected_bar_index])) # 这个 bar 可以增加的高度, 这里 bar 的高度范围只能是 86
        minus = np.random.choice(list(range(0, minus))) # 随机需要增加的长度
    else:
        minus = height_minus
    height_list[selected_bar_index] += minus # 新的 bar height
    
    # 增加他的高度
    selected_bar_position = line_index2position[selected_bar_index] # 确定是哪一个 bar 来进行修改
    height_int = int(height_list[selected_bar_index]) # 转换为 int 类型
    image[99-height_int:99-height_int+minus, selected_bar_position[0]] = 1
    image[99-height_int:99-height_int+minus, selected_bar_position[1]] = 1

    dots_height = (height_list[left_dot_index], height_list[right_dot_index]) # 新的两个 bar height

    return image, min(dots_height)/max(dots_height)


def change_numpy2addVerticalLines_withTopBar(image_numpy_file, height_numpy_file, dots_positions_numpy_file, height_minus_list=None, bar_index_list=None):
    """将原始的 image 转换为 移动点的 image
    """
    image_numpy_data = np.load(image_numpy_file) # 获得 bar image 的 numpy 文件
    bar_height_data = np.load(height_numpy_file) # 获得 bar height 的 numpy 文件
    dots_positions_numpy_data = np.load(dots_positions_numpy_file) # dot position 的 numpy 文件

    new_ratio_list = np.zeros((len(image_numpy_data), 1)) # 存储 bar height 修改过后的 ratio

    for image_index, image in enumerate(image_numpy_data):
        if (height_minus_list == None) and (bar_index_list == None):
            new_image, new_ratio = add_vertiacl_line_withTopbar(im=image[0], 
                                                    height_list=bar_height_data[image_index],
                                                    dots_positions=dots_positions_numpy_data[image_index]) # 这里原始图像大小为 1*100*100, 需要变为 100*100
        else:
            new_image, new_ratio = add_vertiacl_line_withTopbar(im=image[0], 
                                                    height_list=bar_height_data[image_index],
                                                    dots_positions=dots_positions_numpy_data[image_index],
                                                    height_minus=height_minus_list[image_index],
                                                    bar_index=bar_index_list[image_index]) # 这里原始图像大小为 1*100*100, 需要变为 100*100            

        image_numpy_data[image_index][0] = new_image # 修改原始的 image 图像
        new_ratio_list[image_index] = new_ratio # 增加新计算的 ratio

    image_tensor = torch.from_numpy(image_numpy_data)
    label_tensor = torch.from_numpy(new_ratio_list)        

    logger.info('将原始数据集修改为 add vertical lines.')
    return image_tensor, label_tensor

if __name__ == "__main__":
    image_numpy_file = './ratioRegression/exp_output/dataset/fixed_bottom/0/val-image.npy'
    height_numpy_file = './ratioRegression/exp_output/dataset/fixed_bottom/0/val-bar_height.npy'
    dots_positions_numpy_file = './ratioRegression/exp_output/dataset/fixed_bottom/0/val-dots_positions.npy'
    image_tensor, label_tensor = change_numpy2addVerticalLines_withTopBar(image_numpy_file, height_numpy_file, dots_positions_numpy_file)

    # 将 tensor 转换为 np.array
    save_path = './ratioRegression/editBar/example/add_vertical_line_withTopbar'
    os.makedirs(save_path, exist_ok=True)
    image_indexs = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 10000]
    for image_index in image_indexs:
        convert_image = image_tensor.numpy()[image_index][0] # 修改后的图片
        raw_image = np.load(image_numpy_file)[image_index][0] # 原始的图片
        ratio = label_tensor.numpy()[image_index][0]

        # 保存图像
        save_ratio_image(convert_image, os.path.join(save_path, '{}_convert_image_{:.2f}.jpg'.format(image_index, ratio)))
        save_ratio_image(raw_image, os.path.join(save_path, '{}_raw_image.jpg'.format(image_index)))
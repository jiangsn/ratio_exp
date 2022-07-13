'''
@Author: WANG Maonan
@Date: 2021-07-20 03:19:50
@Description: 给定一个 bar image (type 1), 返回每一个的 bar height
@LastEditTime: 2021-08-16 13:41:09
'''
import numpy as np
from ratioReg.TrafficLog.setLog import logger

def calc_type1_bar_heights(im):
    """给定 bar image, 给出每个 bar height;
    通过最顶端的一个 dot 来计算 bar-height

    Args:
        im (array): 需要计算 bar height 的图像
    """
    assert len(im.shape) == 2 # 确保图像是二维的
    
    height_list = [0]*10
    line_index2position = {
                        0:(2,9), 1:(11,18), 2:(20,27), 3:(29,36), 4:(38,45), 
                        5:(55,62), 6:(64,71), 7:(73,80), 8:(82,89), 9:(91,98)
                        } # 根据 bar index 得到 vertical line 的坐标
    
    # 计算 bar height, 通过顶点的位置
    # import ipdb; ipdb.set_trace()
    for i,j in line_index2position.items():
        # height_list[i] = np.sum(im[:, j[0]]) # 通过求和的方式获得 bar height
        # 通过最顶端的顶点的方式获得 bar height
        left_height = max(100 - np.where(im[:, j[0]] == 1)[0], default=0)
        right_height = max(100 - np.where(im[:, j[1]] == 1)[0], default=0)
        if left_height >= right_height:
            height_list[i] = left_height
        else:
            height_list[i] = right_height
    return height_list
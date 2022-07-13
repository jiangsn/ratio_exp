'''
@Author: WANG Maonan
@Date: 2021-08-10 23:30:59
@Description: 给定两个 bar height, 比较他们是否相等
LastEditTime: 2021-12-09 11:42:14
'''
import numpy as np
from ratioReg.TrafficLog.setLog import logger

def compare_bar_height(bar_height1, bar_height2):
    """比较两个 bar height 是否相同

    Args:
        bar_height1 (list): 第一个 bar height, 例如
        bar_height2 (list): 第二个 bar height, 会与第一个进行比较, 例如 
    """
    bar_height_minus = [i-j for i,j in zip(bar_height1, bar_height2)] # 将两个 bar height 做差, 求出相差多少
    logger.info('\n=>bar height 1: {};\n=>bar height 2：{}.\n=>两者的差: {}'.format(bar_height1, bar_height2, bar_height_minus))
    # assert np.mean(bar_height_minus) == 0
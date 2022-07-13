'''
@Author: WANG Maonan
@Date: 2021-06-10 11:10:09
@Description: 给定十个 bar height, 和 两个 black dots 的位置, 可以生成指定的图像
@LastEditTime: 2021-07-24 10:02:44
'''
import numpy as np
import skimage.draw


def data_to_custom_bar(height_data, position):
    """fixed black-dot, and black-dot in the bottom
    """
    barchart = np.zeros((100, 100), dtype=np.bool)

    all_values = [0] * 10 # 高度
    for i in range(10):
        all_values[i] = height_data[i]

    start = 0
    dots_positions = [] # 记录 black-dots 的位置, 类似 [[96, 14], [96, 23]] 的结果
    for i, d in enumerate(all_values):
        if i == 0:
            start += 2
        elif i == 5:
            start += 8
        else:
            start += 0

        gap = 2
        b_width = 7

        left_bar = start + i*gap + i*b_width # 计算左侧 bar 的位置
        right_bar = start + i*gap + i*b_width + b_width # 计算右侧 bar 的位置


        rr, cc = skimage.draw.line(99, left_bar, 99-int(d)+1, left_bar)
        barchart[rr, cc] = 1
        rr, cc = skimage.draw.line(99, right_bar, 99-int(d)+1, right_bar)
        barchart[rr, cc] = 1
        rr, cc = skimage.draw.line(
            99-int(d)+1, left_bar, 99-int(d)+1, right_bar)
        barchart[rr, cc] = 1

        if i == position[0] or i == position[1]: # 那两个 bar 需要黑点
            # place dot here
            barchart[96:97, left_bar+b_width//2:left_bar+b_width//2+1] = 1
            dots_positions.append([96, left_bar+b_width//2])

    return barchart, all_values, dots_positions
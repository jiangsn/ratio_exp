'''
@Author: WANG Maonan
@Date: 2021-07-26 05:02:15
@Description: 根据 csv 文件, 保存所有 badcase 的情况
@LastEditTime: 2021-08-10 23:39:54
'''
import os
import numpy as np
import pandas as pd

from ratioReg.TrafficLog.setLog import logger
from ratioReg.utils.save_image import save_ratio_image
from ratioReg.utils.calcBarHeight_Type1 import calc_type1_bar_heights

def badcase_analysis(root_path, loss_type='loss', threshold=0.08):
    os.makedirs(os.path.join(root_path, 'badcases'), exist_ok=True)
    csv_path = os.path.join(root_path, 'result.csv')
    image_path = os.path.join(root_path, 'image.npy')

    csv_data = pd.read_csv(csv_path)
    image_data = np.load(image_path)

    csv_data['loss'] = np.abs(csv_data['predict'] - csv_data['ground_truth'])
    csv_data['loss_o'] = np.abs(csv_data['predict'] - csv_data['original_ground_truth'])

    for image, (_, row_info) in zip(image_data, csv_data.iterrows()):
        heights = calc_type1_bar_heights(image[0]) # 计算 bar chart 的 bar-heights
        left_bar_height = heights[1] # 左侧 bar-height
        right_bar_height = heights[2] # 右侧 bar-height

        if row_info[loss_type] > threshold:
            logger.info('heights: {}; ground truth: {}; prediction: {}.'.format(heights, row_info['ground_truth'], row_info['predict']))
            save_ratio_image(image[0], '{}/badcases/{}_{}__{:.4f}_{:.4f}.png'.format(root_path, left_bar_height, right_bar_height, row_info['ground_truth'], row_info['predict']))


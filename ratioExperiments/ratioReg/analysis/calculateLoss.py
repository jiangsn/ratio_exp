'''
@Author: WANG Maonan
@Date: 2021-07-18 23:57:41
@Description: 计算 loss（ground truth - prediction）
@LastEditTime: 2021-07-27 08:59:47
'''
import os
import logging
import pandas as pd
import numpy as np

def calculate_loss(data_csv:str, outputFile:str):
    """根据 data_csv 文件计算 loss

    Args:
        data_csv (str): 保存实验结果的 csv 文件 (包含 ground truth 和 predict)
        outputFile (str): 最终保存的文件, 具体到文件夹的路径
    """
    logger = logging.getLogger(__name__)
    os.makedirs(outputFile, exist_ok=True) # 没有就创建文件夹

    data = pd.read_csv(data_csv) # 读取结果
    mae = abs(data['predict']-data['ground_truth']).mean()
    mse = ((data['predict']-data['ground_truth'])**2).mean()
    rmse = np.sqrt(mse)

    # 将 loss 写入文件
    with open('{}/loss.txt'.format(outputFile), 'w') as f:
        f.write(f'rmse={str(rmse)}\n')
        f.write(f'mae={str(mae)}\n')
    logger.info('rmse: {}'.format(rmse))

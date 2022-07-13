'''
@Author: WANG Maonan
@Date: 2021-06-11 14:54:35
@Description: 将 numpy 数据转换为 tensor
@LastEditTime: 2021-12-22 21:49:34
'''
import logging
import torch
import numpy as np


def tensor_loader(numpy_file:str):
    """读取处理好的 npy 文件, 并返回 pytorch 训练使用的 dataloader 数据
    Args:
        pcap_file (str): pcap 文件转换得到的 npy 文件的路径
    """
    logger = logging.getLogger(__name__)
    
    numpy_data = np.load(numpy_file) # 获得 numpy 文件
    numpy_data = torch.from_numpy(numpy_data) # 将 npy 数据转换为 tensor 数据
    logger.info('导入 Tensor 数据, 数据大小, {};'.format(numpy_data.shape))

    return numpy_data
'''
@Author: WANG Maonan
@Date: 2021-05-17 20:58:49
@Description: 读入 npy 文件, 返回 dataloader 数据集
@LastEditTime: 2021-12-22 23:42:05
'''
import logging
import torch
import numpy as np

def data_loader(image_file:str, label_file:str, batch_size:int=256, workers:int=1, pin_memory:bool=True):
    """读取处理好的 npy 文件, 并返回 pytorch 训练使用的 dataloader 数据
    Args:
        image_file (str): pcap 文件转换得到的 npy 文件的路径
        statistic_file (str): 统计特征对应的 npy 文件路径
        label_file (str): 上面的 pcap 文件对应的 label 文件的 npy 文件的路径
        trimed_file_len (int): pcap 被裁剪成的长度
        batch_size (int, optional): 默认一个 batch 有多少数据. Defaults to 256.
        workers (int, optional): 处理数据的进程的数量. Defaults to 1.
        pin_memory (bool, optional): 锁页内存, 如果内存较多, 可以设置为 True, 可以加快 GPU 的使用. Defaults to True.
    Returns:
        DataLoader: pytorch 训练所需要的数据
    """
    logger = logging.getLogger(__name__)
    
    # 载入 npy 数据
    image_data = np.load(image_file) # 获得 image 文件
    label_data = np.load(label_file) # 获得 label 数据

    # 将 npy 数据转换为 tensor 数据
    image_data = torch.from_numpy(image_data)
    label_data = torch.from_numpy(label_data)
    logger.info('Image 文件大小, {}; label 文件大小: {}'.format(image_data.shape, label_data.shape))
    
    # 将 tensor 数据转换为 Dataset->Dataloader
    res_dataset = torch.utils.data.TensorDataset(image_data, label_data) # 合并数据
    res_dataloader = torch.utils.data.DataLoader(
        dataset=res_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=workers        # set multi-work num read data
    )

    return res_dataloader
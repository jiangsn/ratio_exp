'''
@Author: WANG Maonan
@Date: 2021-05-18 01:26:51
@Description: 训练模块
LastEditTime: 2021-12-22 11:04:32
'''
import os
import json
import logging
import numpy as np

import torch
from torch import optim

from ..models.vgg import vgg11, vgg11_bn, vgg19, vgg19_bn
from ..models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from ..models.resnet_bn import resnet18_bn
from ..models.vit_model import ViT_Regression

from .tensorData.dataloader import data_loader
from .tensorData.tensorloader import tensor_loader
from ..utils.helper import adjust_learning_rate, save_checkpoint
from .process import train_process, validate_process, sample_process


def train_model(
    cuda_index: str = 1,
    model_name: str = 'resnet18',
    patch_size: int = 25,
    data_path: str = None,
    lr: float = 0.0001,
    batch_size: int = 256,
    epochs: int = 100,
    output_path: str = None,
):
    """一个模型的训练过程
    """
    logger = logging.getLogger(__name__)
    device = torch.device("cuda:{}".format(cuda_index) if torch.cuda.is_available() else "cpu")
    logger.info('是否使用 GPU 进行训练, {}'.format(device))

    # load model
    logger.info('加载 {} 模型.'.format(model_name))
    if model_name == 'vgg11':
        model = vgg11(pretrained=False, num_classes=1).to(device)  # 定义模型, 分类个数是 1, 也就是 regression
    elif model_name == 'vgg11_bn':  # 带有 BN 的 VGG11
        model = vgg11_bn(pretrained=False, num_classes=1).to(device)
    elif model_name == 'vgg19':
        model = vgg19(pretrained=False, num_classes=1).to(device)
    elif model_name == 'vgg19_bn':
        model = vgg19_bn(pretrained=False, num_classes=1).to(device)
    elif model_name == 'resnet18':  # 去除了 BN 的 ResNet
        model = resnet18(pretrained=False, num_classes=1).to(device)
    elif model_name == 'resnet34':
        model = resnet34(pretrained=False, num_classes=1).to(device)
    elif model_name == 'resnet50':
        model = resnet50(pretrained=False, num_classes=1).to(device)
    elif model_name == 'resnet101':
        model = resnet101(pretrained=False, num_classes=1).to(device)
    elif model_name == 'resnet152':
        model = resnet152(pretrained=False, num_classes=1).to(device)
    elif model_name == 'resnet18_bn':  # 保留 BN 的 ResNet
        model = resnet18_bn(pretrained=False, num_classes=1).to(device)
    elif model_name == 'vit':
        model = ViT_Regression(patch_size=patch_size).to(device)
    else:
        logger.error('模型名称错误.')
        return

    logger.debug(model)  # 打印模型结构
    # 训练集, 测试集, 验证集的路径
    train_image = os.path.join(data_path, 'train-image.npy')
    train_label = os.path.join(data_path, 'train-label.npy')

    val_image = os.path.join(data_path, 'val-image.npy')
    val_label = os.path.join(data_path, 'val-label.npy')

    exp_image = os.path.join(data_path, 'example-image.npy')
    exp_label = os.path.join(data_path, 'example-label.npy')
    logger.info('训练集数据, {}'.format(train_image))

    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()  # MSE
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # load dataset
    train_loader = data_loader(
        image_file=train_image, label_file=train_label, batch_size=batch_size
    )  # 获得 train dataloader
    val_loader = data_loader(
        image_file=val_image, label_file=val_label, batch_size=batch_size
    )  # 获得 val dataloader
    exp_data_tensor, exp_label_tensor = (
        tensor_loader(exp_image),
        tensor_loader(exp_label),
    )  # 加载打印 9 个样例数据集
    logger.info('成功加载数据集.')

    loss_info = {'train': [], 'val': [], 'best_val': dict()}  # 保存「训练」和「测试」过程中的 loss
    best_loss = 999999
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, lr)  # 动态调整学习率
        train_loss = train_process(
            train_loader, model, criterion, optimizer, epoch, device, 100, model_name
        )  # train for one epoch
        val_loss = validate_process(
            val_loader, model, criterion, device, 50, model_name
        )  # evaluate on validation set
        sample_process(model, exp_data_tensor, exp_label_tensor, device)  # 跑一下模型在 9 个值上的预测结果
        # 将 loss 从 mse -> rmse
        train_loss = np.sqrt(train_loss)
        val_loss = np.sqrt(val_loss)
        # 记录 loss 的变化
        loss_info['train'].append(train_loss)  # train 上面 loss 的变化
        loss_info['val'].append(val_loss)  # val 上面 loss 的变化

        # 保存最优的模型
        is_best = val_loss < best_loss  # 如果 val loss 小于最优的 loss
        best_loss = min(val_loss, best_loss)  # 更新 best loss
        if is_best:
            loss_info['best_val']['val'] = val_loss
            loss_info['best_val']['train'] = train_loss
        os.makedirs(output_path, exist_ok=True)
        save_checkpoint(
            {'state_dict': model.state_dict()}, is_best, os.path.join(output_path, model_name)
        )

    # 将关于 loss 的信息保存为 json 文件
    loss_name = 'loss_{}.json'.format(model_name)
    with open(os.path.join(output_path, loss_name), "w") as f:
        json.dump(loss_info, f, indent=4)

    logger.info('Finished! (*￣︶￣)\n\n')

'''
@Author: WANG Maonan
@Date: 2021-05-17 23:44:31
@Description: 模型训练的流程, 这里是一个 epoch 的训练流程
@LastEditTime: 2021-12-10 01:50:40
'''
import logging
from ...utils.helper import AverageMeter


def train_process(train_loader, model, criterion, optimizer, epoch:int, device:str, print_freq:int, model_name:str):
    """训练一个 epoch 的流程
    Args:
        train_loader (dataloader): [description]
        model ([type]): [description]
        criterion ([type]): [description]
        optimizer ([type]): 优化器
        epoch (int): 当前所在的 epoch
        device (torch.device): 是否使用 gpu
        print_freq (int): 打印频率
    """
    logger = logging.getLogger(__name__)

    losses = AverageMeter()  # 在一个 train loader 中的 loss 变化

    model.train()  # 切换为训练模型

    for i, (image, target) in enumerate(train_loader):
        image = image.to(device)
        target = target.to(device)
        output = model(image)  # 得到模型预测结果
        loss = criterion(output, target)  # 计算 loss

        # 记录 loss
        losses.update(loss.item(), image.size(0))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % print_freq == 0: # 打印日志信息
            logger.info('Model, {0}; Train Epoch: [{1}][{2}/{3}], Loss {loss.val:.6f} ({loss.avg:.6f})'.format(
                model_name, epoch, i, len(train_loader), loss=losses))
    
    return losses.avg # 返回一个 epoch 中的 loss 的平均值

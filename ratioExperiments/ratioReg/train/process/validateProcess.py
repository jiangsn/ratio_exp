'''
@Author: WANG Maonan
@Date: 2021-05-18 05:29:51
@Description: 数据集验证模块
@LastEditTime: 2021-06-08 18:13:56
'''
import logging
import torch
from ...utils.helper import AverageMeter

def validate_process(val_loader, model, criterion, device, print_freq, model_name):
    """对测试集进行验证, 并保存最终的结果
    """
    logger = logging.getLogger(__name__)
    losses = AverageMeter()
    
    model.eval()  # switch to evaluate mode

    for i, (image, target) in enumerate(val_loader):
        
        image = image.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(image)  # compute output
            loss = criterion(output, target)  # 计算验证集的 loss

            # measure accuracy and record loss
            losses.update(loss.item(), image.size(0))

            if (i+1) % print_freq == 0:
                logger.info('Model, {0}; Test Epoch: [{1}/{2}], Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                    model_name, i, len(val_loader), loss=losses))

    logger.info(' * Loss {loss.val:.6f} ({loss.avg:.6f})'.format(loss=losses))
    
    return losses.avg # 返回 loss 的平均值
'''
@Author: WANG Maonan
@Date: 2021-06-15 13:41:20
@Description: 使用训练好的模型, 预测 test, 将结果保存为 csv
LastEditTime: 2021-12-22 11:00:57
'''
import os
import torch
import numpy as np
import pandas as pd
import logging

from ..models.vgg import vgg11, vgg11_bn, vgg19, vgg19_bn
from ..models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from ..models.resnet_bn import resnet18_bn
from ..models.vit_model import ViT_Regression

from .tensorData.tensorloader import tensor_loader

def get_predictions(
                    model_name:str=None,
                    model_path:str=None,
                    image_path:str=None,
                    label_path:str=None,
                    height_path:str=None,
                    dots_path:str=None,
                    image_resolution:int=None,
                    patch_size:int=None
                    ):
    """加载模型, 并在测试集上进行测试, 将预测结果与真实结果以 Dataframe 的形式进行返回

    Args:
        model_name (str): model 的名称
        model_path (str): model 的文件路径
        image_path (str): image 的路径
        label_path (str): label 的路径
        height_path (str): bar-heights 的路径
        dots_path (str): dots 的路径, 记录 black-dots 的位置
    """
    logger = logging.getLogger(__name__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dots_position2index = {5:0, 14:1, 23:2, 32:3, 41:4, 58:5, 67:6, 76:7, 85:8, 94:9} # 从 dots-position 定位到 bar index

    logger.info('正在使用, {}'.format(device))
    logger.info('正在分析, model_path:{}, image_path:{}'.format(model_path, image_path))
    
    # 导入模型
    if model_name == 'vgg11':
        logger.info('使用 vgg11 模型.')
        model = vgg11(pretrained=False, num_classes=1).to(device)  # 定义模型, 分类个数是 1, 也就是 regression
    elif model_name == 'vgg11_bn':
        logger.info('使用 vgg11_bn 模型.')
        model = vgg11_bn(pretrained=False, num_classes=1).to(device)
    elif model_name == 'vgg19':
        logger.info('使用 vgg19 模型.')
        model = vgg19(pretrained=False, num_classes=1).to(device)
    elif model_name == 'vgg19_bn':
        logger.info('使用 vgg19_bn 模型 (带有 batch normalization).')
        model = vgg19_bn(pretrained=False, num_classes=1).to(device)
    elif model_name == 'resnet18':  # 去除了 BN 的 ResNet
        logger.info('使用 resnet18 模型.')
        model = resnet18(pretrained=False, num_classes=1).to(device)
    elif model_name == 'resnet34': 
        logger.info('使用 resnet34 模型.')
        model = resnet34(pretrained=False, num_classes=1).to(device)
    elif model_name == 'resnet50':
        logger.info('使用 resnet50 模型.')
        model = resnet50(pretrained=False, num_classes=1).to(device)
    elif model_name == 'resnet101':
        logger.info('使用 resnet101 模型.')
        model = resnet101(pretrained=False, num_classes=1).to(device)
    elif model_name == 'resnet152':
        logger.info('使用 resnet152 模型.')
        model = resnet152(pretrained=False, num_classes=1).to(device)
    elif model_name == 'resnet18_bn':  # 保留 BN 的 ResNet
        logger.info('使用 resnet18_bn 模型.')
        model = resnet18_bn(pretrained=False, num_classes=1).to(device)
    elif model_name == 'vit':
        logger.info('使用 vit 模型.')
        model = ViT_Regression(patch_size=patch_size).to(device)
    else:
        logger.error('模型名称错误.')
        return

    checkpoint = torch.load(model_path) # 加载模型
    model.load_state_dict(checkpoint['state_dict']) # 加载模型参数
    model.eval() # 转换为测试模式(!!!重要)

    # 导入测试数据
    test_data_tensor, test_label_tensor = tensor_loader(image_path), tensor_loader(label_path)
    test_height_tensor, test_dots_tensor = tensor_loader(height_path), tensor_loader(dots_path)
    test_data_tensor, test_label_tensor = test_data_tensor.to(device), test_label_tensor.to(device) # 将 tensor 放在 cpu 或是 gpu

    # 得到预测值与真实值
    predict_list = [] # 存储模型的预测值
    ground_truth_list = [] # 新的 label
    low_bar_height_list = [] # 长的 bar
    high_bar_height_list = [] # 短的 bar
    for i, j, _bar_height, _dots in zip(test_data_tensor, test_label_tensor, test_height_tensor, test_dots_tensor): # 逐个数据进行预测
        predict_item = model(i.view(1, 1, image_resolution, image_resolution)).item() # 得到模型预测值
        predict_list.append(predict_item) # 预测值
        ground_truth_list.append(j.item()) # ground truth

        _leftbar_index, rightbar_index = [dots_position2index[dots_position[1].cpu().item()] for dots_position in _dots]
        _bar_height_list = [_bar_height[_leftbar_index].cpu().item(), _bar_height[rightbar_index].cpu().item()]
        low_bar_height_list.append(min(_bar_height_list))
        high_bar_height_list.append(max(_bar_height_list))

    
    model_result = pd.DataFrame(
        {
            'lowerbarHeight': low_bar_height_list, 
            'higherbarHeight': high_bar_height_list,
            'predict': predict_list, 
            'ground_truth': ground_truth_list
        }
    )
    return model_result, test_data_tensor.cpu(), test_height_tensor.cpu(), test_dots_tensor.cpu()

def predictions_inDataset(
                        model_name:str=None, 
                        model_path:str=None,
                        data_path:str=None,
                        image_resolution:int=100,
                        patch_size:int=25,
                        output_path:str=None
                        ):
    """测试模型在测试集上的好坏
    """
    logger = logging.getLogger(__name__)
    
    logger.info('正在评价模型 {}/{}'.format(model_path, model_name))

    test_image = os.path.join(data_path, 'test-image.npy')
    test_label = os.path.join(data_path, 'test-label.npy')
    test_bar_height = os.path.join(data_path, 'test-bar_height.npy') # 每个 bar 的高度
    test_dots_positions = os.path.join(data_path, 'test-dots_positions.npy') # black-dot 的位置

    model_result, image_data, bar_height, dots_position = get_predictions(model_name=model_name,
                                                                        model_path=os.path.join(model_path, model_name), 
                                                                        image_path=test_image, 
                                                                        label_path=test_label, 
                                                                        height_path=test_bar_height, 
                                                                        dots_path=test_dots_positions,
                                                                        image_resolution=image_resolution,
                                                                        patch_size=patch_size)
    # 将 image data 进行保存
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, 'image.npy'), image_data.numpy()) # 保存图像
    np.save(os.path.join(output_path, 'bar_height.npy'), bar_height.numpy()) # 保存 bar 高度
    np.save(os.path.join(output_path, 'dots_positions.npy'), dots_position.numpy()) # 保存 dots 的位置
    model_result.to_csv(os.path.join(output_path, 'evaluate_result.csv'), index=False) # 将结果保存至 csv 文件
    logger.info('模型 {} 分析完毕, 数据保存至, {}.\n\n'.format(model_name, output_path))  
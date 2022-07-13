'''
@Author: WANG Maonan
@Date: 2021-06-15 13:41:20
@Description: 使用训练好的模型, 预测测试集, 将结果保存为 csv
LastEditTime: 2021-12-22 11:00:57
'''
import os
import torch
import numpy as np
import pandas as pd
import logging

from ratioReg.models.vgg import vgg11, vgg11_bn
from ratioReg.models.resnet import resnet18
from ratioReg.models.resnet_bn import resnet18_bn
from ratioReg.models.vit_model import ViT_Regression

from ratioReg.train.tensorData.tensorloader import tensor_loader

from ratioReg.editBar.remove_dots import change_numpy2removedots
from ratioReg.editBar.move_dots import change_numpy2movedots
from ratioReg.editBar.remove_horizontal_line import change_numpy2removeHorizontalLine
from ratioReg.editBar.add_vertical_line import change_numpy2addVerticalLines
from ratioReg.editBar.change_barTop_positions import change_numpy2changeBarTopPosition
from ratioReg.editBar.add_vertical_line_withTopbar import change_numpy2addVerticalLines_withTopBar
from ratioReg.editBar.remove_vertical_upper_line import change_numpy2removeVerticalUpperLines
from ratioReg.editBar.remove_vertical_lower_line import change_numpy2removeVerticalLowerLines
from ratioReg.editBar.remove_vertical_lines import change_numpy2removeVerticalLine
from ratioReg.editBar.remove_vertical_middle_line import change_numpy2removeVerticalMiddleLines
from ratioReg.editBar.keep_left_barHeight import change_numpy2keepleftbarHeight
from ratioReg.editBar.keep_right_barHeight import change_numpy2keeprightbarHeight
from ratioReg.editBar.remove_random_vertical_line import change_numpy2removeRandomVerticalLine
from ratioReg.editBar.remove_random_double_vertical_lines import change_numpy2removeRandomDoubleVerticalLine


def get_predictions(model_name, model_path, image_path, label_path, height_path, dots_path, test_type, image_resolution, patch_size):
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

    logger.info('正在使用, {}'.format(device))
    logger.info('正在分析, model_path:{}, image_path:{}'.format(model_path, image_path))
    
    # 导入模型
    if model_name == 'vgg11':
        logger.info('使用 vgg11 模型.')
        model = vgg11(pretrained=False, num_classes=1).to(device)  # 定义模型, 分类个数是 1, 也就是 regression
    elif model_name == 'vgg11_bn':
        logger.info('使用 vgg11_bn 模型.')
        model = vgg11_bn(pretrained=False, num_classes=1).to(device)
    elif model_name == 'resnet18':  # 去除了 BN 的 ResNet
        logger.info('使用 resnet18 模型.')
        model = resnet18(pretrained=False, num_classes=1).to(device)
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

    original_test_label_tensor = tensor_loader(label_path) # 原始测试集的 label
    # 导入测试数据
    if test_type == 'original':
        logger.info('现在测试数据集为 origin.')
        test_data_tensor, test_label_tensor = tensor_loader(image_path), tensor_loader(label_path)
    elif test_type == 'remove_dots':
        logger.info('现在测试数据集为 remove-dots.') # 将 dots 删除的实验
        test_data_tensor = change_numpy2removedots(image_numpy_file=image_path, dots_positions_numpy_file=dots_path)
        test_label_tensor = tensor_loader(label_path)
    elif test_type == 'move_dots':
        logger.info('现在测试数据集为 move-dots.') # 移动 dots 位置的实验
        test_data_tensor, test_label_tensor, _ = change_numpy2movedots(image_numpy_file=image_path, height_numpy_file=height_path, dots_positions_numpy_file=dots_path)
    elif test_type == 'remove_horizontal_lines':
        logger.info('现在测试数据集为 remove-horizontal-lines.') # 去除水平线
        test_data_tensor, _ = change_numpy2removeHorizontalLine(image_numpy_file=image_path, height_numpy_file=height_path, dots_positions_numpy_file=dots_path)
        test_label_tensor = tensor_loader(label_path)
    elif test_type == 'remove_vertical_lines': # 去除全部垂直线
        logger.info('现在测试数据集为 remove-vertical-lines.') 
        test_data_tensor, _ = change_numpy2removeVerticalLine(image_numpy_file=image_path, height_numpy_file=height_path, dots_positions_numpy_file=dots_path)
        test_label_tensor = tensor_loader(label_path)
    elif test_type == 'change_barTop_positions': # 修改 bar-top 的位置, 将 bar-top 向下移动
        logger.info('现在测试数据集为 change_barTop_positions.') 
        test_data_tensor, test_label_tensor, _ = change_numpy2changeBarTopPosition(image_numpy_file=image_path, height_numpy_file=height_path, dots_positions_numpy_file=dots_path)
    elif test_type == 'add_vertical_lines':
        logger.info('现在测试数据集为 add-vertical-lines.') # 增加部分垂直线, 去除 top-bar
        test_data_tensor, test_label_tensor = change_numpy2addVerticalLines(image_numpy_file=image_path, height_numpy_file=height_path, dots_positions_numpy_file=dots_path)
    elif test_type == 'add_vertical_lines_withTopbar':
        logger.info('现在测试数据集为 add-vertical-lines-with-TopBar.') # 增加部分垂直线, 去除 top-bar
        test_data_tensor, test_label_tensor = change_numpy2addVerticalLines_withTopBar(image_numpy_file=image_path, height_numpy_file=height_path, dots_positions_numpy_file=dots_path)
    elif test_type == 'remove_vertical_upper_lines':
        logger.info('现在测试数据集为 remove-vertical-upper-lines.') # 减少部分垂直线 (上端)
        test_data_tensor, test_label_tensor = change_numpy2removeVerticalUpperLines(image_numpy_file=image_path, height_numpy_file=height_path, dots_positions_numpy_file=dots_path)
    elif test_type == 'remove_vertical_lower_lines':
        logger.info('现在测试数据集为 remove-vertical-lower-lines.') # 减少部分垂直线 (下端)
        test_data_tensor, test_label_tensor = change_numpy2removeVerticalLowerLines(image_numpy_file=image_path, height_numpy_file=height_path, dots_positions_numpy_file=dots_path)
    elif test_type == 'remove_vertical_middle_lines':
        logger.info('现在测试数据集为 remove-vertical-middle-lines.') # 减少部分垂直线 (中间)
        test_data_tensor, test_label_tensor = change_numpy2removeVerticalMiddleLines(image_numpy_file=image_path, height_numpy_file=height_path, dots_positions_numpy_file=dots_path)
    elif test_type == 'keep_left_barHeight':
        logger.info('现在测试数据集为 keep_left_barHeight.') # 只保留左侧的 bar-height
        test_data_tensor = change_numpy2keepleftbarHeight(image_numpy_file=image_path, height_numpy_file=height_path, dots_positions_numpy_file=dots_path)
        test_label_tensor = tensor_loader(label_path)
    elif test_type == 'keep_right_barHeight':
        logger.info('现在测试数据集为 keep_right_barHeight.') # 只保留左侧的 bar-height
        test_data_tensor = change_numpy2keeprightbarHeight(image_numpy_file=image_path, height_numpy_file=height_path, dots_positions_numpy_file=dots_path)
        test_label_tensor = tensor_loader(label_path)      
    elif test_type == 'remove_random_one_vertical_line':
        logger.info('现在测试数据集为 remove_random_one_vertical_line.') # 随机删除一个竖线
        test_data_tensor = change_numpy2removeRandomVerticalLine(image_numpy_file=image_path, height_numpy_file=height_path, dots_positions_numpy_file=dots_path)
        test_label_tensor = tensor_loader(label_path)        
    elif test_type == 'remove_random_double_vertical_lines':
        logger.info('现在测试数据集为 remove_random_double_vertical_lines.') # 随机删除一个竖线
        test_data_tensor = change_numpy2removeRandomDoubleVerticalLine(image_numpy_file=image_path, height_numpy_file=height_path, dots_positions_numpy_file=dots_path)
        test_label_tensor = tensor_loader(label_path)
    # 将 tensor 放在 cpu 或是 gpu
    test_data_tensor, test_label_tensor = test_data_tensor.to(device), test_label_tensor.to(device)

    # 得到预测值与真实值
    predict_list = [] # 存储模型的预测值
    ground_truth_list = [] # 新的 label
    original_ground_truth_list = [] # 原始的 label
    for i,j,k in zip(test_data_tensor, test_label_tensor, original_test_label_tensor): # 逐个数据进行预测
        predict_item = model(i.view(1, 1, image_resolution, image_resolution)).item() # 得到模型预测值
        predict_list.append(predict_item)
        ground_truth_list.append(j.item())
        original_ground_truth_list.append(k.item())
    
    model_result = pd.DataFrame({'predict':predict_list, 'ground_truth':ground_truth_list, 'original_ground_truth':original_ground_truth_list}) # 这里会有两种 ground truth, ground_truth=>可能是重新计算的, original_ground_truth=>原始的 label
    return model_result, test_data_tensor.cpu()

def predictions_inDataset(cfg):
    """对于一类数据集, 衡量全部十个模型的性能, 并将结果保存在 csv 文件中
    """
    logger = logging.getLogger(__name__)
    
    for k in range(cfg.evaluate.num): # 对 N 个模型进行评估
        logger.info('正在评价模型 {}'.format(k))
        
        model_path = os.path.join(cfg.evaluate.model_path, cfg.evaluate.type, cfg.evaluate.checkpoint_type, cfg.evaluate.model_name) # 模型路径
        model_name = '{}_{}'.format(k, cfg.evaluate.model_name) # 模型名称

        test_image = os.path.join(cfg.evaluate.data_path, cfg.evaluate.type, cfg.evaluate.dataset_type, str(k), 'test-image.npy')
        test_label = os.path.join(cfg.evaluate.data_path, cfg.evaluate.type, cfg.evaluate.dataset_type, str(k), 'test-label.npy')
        test_bar_height = os.path.join(cfg.evaluate.data_path, cfg.evaluate.type, cfg.evaluate.dataset_type, str(k), 'test-bar_height.npy') # 每个 bar 的高度
        test_dots_positions = os.path.join(cfg.evaluate.data_path, cfg.evaluate.type, cfg.evaluate.dataset_type, str(k), 'test-dots_positions.npy') # black-dot 的位置

        model_result, image_data = get_predictions(model_name=cfg.evaluate.model_name,
                                                    model_path=os.path.join(model_path, model_name), 
                                                    image_path=test_image, 
                                                    label_path=test_label, 
                                                    height_path=test_bar_height, 
                                                    dots_path=test_dots_positions,
                                                    test_type=cfg.evaluate.test_type,
                                                    image_resolution=cfg.image_resolution,
                                                    patch_size = cfg.train.patch_size)
        # 将 image data 进行保存
        csv_path = os.path.join(cfg.evaluate.result_path, cfg.evaluate.type, cfg.evaluate.test_type, 'Model_{}_Dataset_{}/{}'.format(cfg.evaluate.checkpoint_type, cfg.evaluate.dataset_type, str(k))) # 文件名为 「模型名称+测试集名称」
        os.makedirs(csv_path, exist_ok=True)
        np.save(os.path.join(csv_path, 'image.npy'), image_data.numpy()) # 保存图像
        model_result.to_csv(os.path.join(csv_path, 'result.csv'), index=False) # 将结果保存至 csv 文件
        logger.info('模型 {} 分析完毕, 数据保存至, {}.\n\n'.format(k, csv_path))
    
    logger.info('所有数据分析完毕!')
        

        
'''
@Author: WANG Maonan
@Date: 2021-07-25 22:49:43
@Description: 分析 loss 的大小与 bar-height 之间的关系
LastEditTime: 2021-12-22 13:02:30
'''
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['science','ieee','no-latex'])

def barHeight_loss_heatmap(
                    data_type:str,
                    data_csv:str,
                    bar_height_file:str,
                    output_path:str
                    ):
    """找出 bar-height 与 loss 之间的关系

    Args:
        data (str): 模型运行的结果集合, evaluate_result.csv 文件 (包含 predict 和 ground truth)
        bar_height_file (str): 存放 bar-height 的文件
        output_path (str): 图片的保存路径, 类似, ./ratioRegression/exp_output/model_performance/
    """
    logger = logging.getLogger(__name__)
    data_type_dots_position = {
        'type1':[1, 2],
        'type2':[0, 4],
        'type3':[1, 6],
        'type4':[3, 7],
        'type5':[2, 3]
    } # 不同类型数据中 black-dots 的位置, 例如 type1 在 1,2 这两个 bar 上面
    dots_positions = data_type_dots_position[data_type]
    os.makedirs(output_path, exist_ok=True) # 没有就创建文件夹
    heatmap_dict = {i:{j:[] for j in range(5, 86)} for i in range(5, 86)} # 记录 bar-height 从 5-86 之间的 loss

    data = pd.read_csv(data_csv) # 加在 predict 和 ground-truth 的结果
    bar_heights = np.load(bar_height_file) # 加在 bar_height 的结果

    for (index, row_info), bar_height in zip(data.iterrows(), bar_heights):
        left_bar_height = bar_height[dots_positions[0]] # 左侧 bar-height, index=1
        right_bar_height = bar_height[dots_positions[1]] # 右侧 bar-height, index=2

        loss = np.abs(row_info['predict'] - row_info['ground_truth'])
        heatmap_dict[left_bar_height][right_bar_height].append(loss)
        
    # 将 dict 数据转换为 list 数据, 方便绘制 heatmap
    heatmap_data = []
    for i in range(5, 86):
        heatmap_data_tmp = []
        for j in range(5, 86):
            loss_list = heatmap_dict.get(i).get(j)
            if len(loss_list) >= 1:
                heatmap_data_tmp.append(np.mean(loss_list)) # loss 是计算平均值的
            else:
                heatmap_data_tmp.append(0)
        heatmap_data.append(heatmap_data_tmp)
        
    # 将图像保存下来
    pd_heatmap_data = pd.DataFrame(heatmap_data, index=list(range(5, 86)), columns=list(range(5, 86)))
    save_heatmap(pd_heatmap_data, output_path) # 保存 heatmap 图像

def save_heatmap(data, imagePath, image_name='barHeight_loss_heatmap.png'):
    """绘制 heatmap 数据

    Args:
        data (list): 绘制 heatmap 的数据
        imagePath (str): 图像保存的文件夹
    """
    plt.figure(figsize = (14,14))
    
    ax = sns.heatmap(data, annot=False, cmap="BuPu")
    # ax = sns.heatmap(data, annot=False, cmap="Greys", vmin=0, vmax=0.5)
    # 设置 y 轴的字体的大小
    plt.title('The Relationship between Bar-height Combination and Loss', fontsize='xx-large')
    # 设置 ticks
    plt.xlabel('Bar-Height', fontsize='xx-large')
    plt.ylabel('Bar-Height', fontsize='xx-large')
    # 翻转坐标轴
    ax.invert_yaxis()
    # 保存图片
    plt.savefig(os.path.join(imagePath, image_name), dpi=500)
    plt.close()
'''
@Author: WANG Maonan
@Date: 2021-07-18 23:57:41
@Description: 对一种 condition 进行分析, 给出 condition prediction 和 ground truth 之间关系图, 同时绘制出 ratio 与 loss 的关系
@LastEditTime: 2021-07-27 08:59:47
'''
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import logging
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['science','ieee','no-latex'])

def scatterplot_histplot_prediction(data_csv:str, outputFile:str):
    """将结果绘制为 scatterplot 与 histplot, 并保存图像.
    绘制 prediction 与 ground truth.

    Args:
        data_csv (str): 保存实验结果的 csv 文件 (包含 ground truth 和 predict)
        outputFile (str): 最终保存的文件, 具体到文件夹的路径
    """
    logger = logging.getLogger(__name__)
    os.makedirs(outputFile, exist_ok=True) # 没有就创建文件夹

    data = pd.read_csv(data_csv) # 读取结果
    grid = sns.JointGrid(x='ground_truth', y='predict', data=data) # 绘制图像

    grid.plot_joint(sns.scatterplot)
    grid.plot_marginals(
        sns.histplot, 
        bins=[i/100 for i in range(101)],
        kde=True
    ) # 绘制侧边的柱状图
    sns.lineplot([0, 1], [0, 1], linewidth=1, ax=grid.ax_joint) # 绘制对角线

    # 修改 x_label 和 y_label 的名称
    grid.ax_joint.set_xlabel('Ground Truth', fontsize=20)
    grid.ax_joint.set_ylabel('Prediction', fontsize=20)

    # 设置 ticks 的大小
    grid.ax_joint.set_xticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    grid.ax_joint.set_xticklabels((0, 0.2, 0.4, 0.6, 0.8, 1), fontsize=20)
    grid.ax_joint.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    grid.ax_joint.set_yticklabels((0, 0.2, 0.4, 0.6, 0.8, 1), fontsize=20)

    # 设置范围
    grid.ax_joint.set_ylim([-0.1, 1.1])
    grid.ax_joint.set_xlim([-0.1, 1.1])

    plt.savefig('{}/{}.png'.format(outputFile, 'predict-groundTruth'))
    plt.close()
    logger.info('图片生成成功, {}/{}.png'.format(outputFile, 'groundTruth'))

def scatterplot_histplot_loss(data, outputFile):
    """将结果绘制为 scatterplot 与 histplot, 并保存图像 ==> 绘制 loss 与 ground truth.

    Args:
        data (pd.dataframe): 一个 condition 所有 runs 的结果汇总
        outputFile (str): 最终保存的文件, 具体到文件夹的路径
    """
    logger = logging.getLogger(__name__)
    os.makedirs(outputFile, exist_ok=True) # 没有就创建文件夹

    for ground_truth_type, loss_type in {'ground_truth':'loss', 'original_ground_truth':'loss_o'}.items(): # 这里绘制两副图像
        grid = sns.JointGrid(x=ground_truth_type, y=loss_type, data=data) # 绘制图像

        grid.plot_joint(sns.scatterplot)
        grid.plot_marginals(
            sns.histplot, 
            bins=[i/100 for i in range(101)],
            kde=True
        ) # 绘制侧边的柱状图

        # 修改 x_label 和 y_label 的名称
        grid.ax_joint.set_xlabel('Ground Truth', fontsize=20)
        grid.ax_joint.set_ylabel('Loss', fontsize=20)

        # 设置 ticks 的大小
        grid.ax_joint.set_xticks((0, 0.2, 0.4, 0.6, 0.8, 1))
        grid.ax_joint.set_xticklabels((0, 0.2, 0.4, 0.6, 0.8, 1), fontsize=20)
        grid.ax_joint.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
        grid.ax_joint.set_yticklabels((0, 0.2, 0.4, 0.6, 0.8, 1), fontsize=20)

        # 设置范围
        grid.ax_joint.set_ylim([-0.1, 1.1])
        grid.ax_joint.set_xlim([-0.1, 1.1])

        plt.savefig('{}/loss_{}.png'.format(outputFile, ground_truth_type))
        plt.close()
        logger.info('图片生成成功, {}/loss_{}.png'.format(outputFile, ground_truth_type))
'''
@Author: WANG Maonan
@Date: 2021-07-18 23:55:08
@Description: 对结果分析的流程, TODO, 把结果不好的值都保存下来
LastEditTime: 2021-12-22 13:10:17
'''
import os
import pandas as pd

from ratioReg.TrafficLog.setLog import logger
from ratioReg.utils.setConfig import setup_config

from ratioReg.analysis.concatConditionResult import concat_result
from ratioReg.analysis.scatterHistCondition import scatterplot_histplot_prediction, scatterplot_histplot_loss
from ratioReg.analysis.barHeightCombinationsLoss import barHeight_loss_heatmap
from ratioReg.analysis.badcaseAnalysis import badcase_analysis

def analysis_pipeline():
    cfg = setup_config() # 获取 config 文件
    
    # 获得完整的 data
    data = concat_result(result_path = cfg.evaluate.result_path, 
                         overall_type = cfg.evaluate.type,
                         test_type = cfg.evaluate.test_type, 
                         checkpoint_type = cfg.evaluate.checkpoint_type, 
                         dataset_type = cfg.evaluate.dataset_type, 
                         run_nums = cfg.evaluate.num)
    logger.info('所有模型结果合并成功!')
    
    # 绘制 heatmap
    if False:
        barHeight_loss_heatmap(data=data,
                                result_path = cfg.evaluate.result_path, 
                                overall_type = cfg.evaluate.type,
                                test_type = cfg.evaluate.test_type, 
                                checkpoint_type = cfg.evaluate.checkpoint_type, 
                                dataset_type = cfg.evaluate.dataset_type, 
                                run_nums = cfg.evaluate.num)
        logger.info('Loss 与 Bar-Height 分析成功!')

    # 生成 scatter 与 hist 图像
    run_path = os.path.join(cfg.evaluate.result_path, cfg.evaluate.type, cfg.evaluate.test_type, 'Model_{}_Dataset_{}'.format(cfg.evaluate.checkpoint_type, cfg.evaluate.dataset_type)) # 文件名为 「模型名称+测试集名称」
    scatterplot_histplot_prediction(data, run_path) # 绘制总的结果
    scatterplot_histplot_loss(data, run_path)
    logger.info('绘制完毕总的 ScatterPlot 图像!')

    # 将所有的 bad case 进行保存
    loss_type='loss_o'
    for run_nums in range(cfg.evaluate.num): # 分别绘制每一次 run 的结果, 并把 loss 较大的绘制出来
        csv_path = os.path.join(run_path, '{}/result.csv'.format(run_nums))
        csv_data = pd.read_csv(csv_path) # 读取 csv 文件
        scatterplot_histplot_prediction(csv_data, os.path.join(run_path, str(run_nums))) # 每次 run 的结果均绘制 scatter 与 hist 图像
        badcase_analysis(root_path=os.path.join(run_path, str(run_nums)), loss_type=loss_type, threshold=2) # 保存 bad case
        logger.info('分析完毕第 {} 个 csv 文件.'.format(run_nums))
    
    logger.info('所有分析完毕!')
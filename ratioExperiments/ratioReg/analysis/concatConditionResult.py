'''
@Author: WANG Maonan
@Date: 2021-07-18 23:57:41
@Description: 将一种 condition 的 N 次 RUN 的结果组合在一起
@LastEditTime: 2021-09-03 07:43:31
'''
import os
import numpy as np
import pandas as pd

from ratioReg.TrafficLog.setLog import logger

def concat_result(result_path, overall_type, test_type, checkpoint_type, dataset_type, run_nums):
    """将 N 次实验结果合成一个 DataFrame, 最终 dataframe 的列包括, predict, ground_truth, original_ground_truth, runs, loss, loss_o

    Args:
        result_path (str): 模型结果保存的路径, 类似, ./ratioRegression/exp_output/model_performance/
        overall_type (str): 五种类型的数据集, type1, type2, type3, type4, type5
        test_type (str): original, remove_dots, remove_horizontal_lines, remove_vertical_lines, remove_vertical_upper_lines, add_vertical_lines, move_dots
        checkpoint_type (str): 模型的类型, 即模型是使用哪一种数据集训练出来的, fixed_bottom
        dataset_type (str): 测试数据集的类型, fixed_bottom
        run_nums (int): 模型一共 RUN 了多少次, 对应有多少个 csv 文件
    """
    
    data = pd.DataFrame({'predict':[], 'ground_truth':[], 'original_ground_truth':[], 'runs':[]}) # 将 N 次运行结果放在一起
    for k in range(run_nums):
        csv_path = os.path.join(result_path, overall_type, test_type, 'Model_{}_Dataset_{}/{}/result.csv'.format(checkpoint_type, dataset_type, str(k))) # 获得 csv 文件的路径
        csv_data = pd.read_csv(csv_path) # 读取结果
        csv_data['runs'] = str(k) # 加上是第一次 run 的结果
    
        data = pd.concat([data, csv_data]) # 将 N 次结果组合
    
    # 在 data 中添加两个 loss
    data['loss'] = np.abs(data['predict'] - data['ground_truth'])
    data['loss_o'] = np.abs(data['predict'] - data['original_ground_truth'])

    loss = np.mean(np.abs(data['predict'] - data['ground_truth']))
    loss_o = np.mean(np.abs(data['predict'] - data['original_ground_truth']))

    logger.info('\n\nGround Truth Loss, {:.6f}; Original Ground Truth Loss, {:.6f};\n\n\n'.format(loss, loss_o))
    
    return data
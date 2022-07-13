'''
@Author: WANG Maonan
@Date: 2021-05-18 05:31:42
@Description: 总的入口函数
LastEditTime: 2021-07-19 03:40:15
'''
import fire
from ratioReg.entry.train import train_pipeline
from ratioReg.entry.generateData import generate_data_pipeline
from ratioReg.entry.evaluatePipeline import evaluate_pipeline
from ratioReg.entry.analysisPipeline import analysis_pipeline

def help():
    """使用的一些简单说明
    """
    data = '''
    => 数据生成的流程 (先把数据保存为 numpy 的格式)
    python -m ratioRegression generate_data_pipeline
    => 数据训练的流程 (基础模型训练)
    python -m ratioRegression train_pipeline
    => 测试模型性能 (所有结果保存到 csv 中)
    python -m ratioRegression evaluate_pipeline
    => 分析最终结果 (分析结果的 csv 文件)
    python -m ratioRegression analysis_pipeline
    '''
    return data

if __name__ == "__main__":
    fire.Fire()
'''
@Author: WANG Maonan
@Date: 2021-12-22 23:39:35
@Description: 分析模型的输出结果
@LastEditTime: 2021-12-23 00:56:37
'''
from ratioReg.analysis.scatterHistCondition import scatterplot_histplot_prediction
from ratioReg.analysis.barHeightCombinationsLoss import barHeight_loss_heatmap
from ratioReg.analysis.calculateLoss import calculate_loss
from ratioReg.ratioLog.initLog import init_logging
from ratioReg.utils.get_abs_path import getAbsPath

pathConvert = getAbsPath(__file__)
init_logging(log_path=pathConvert('./'), log_level=0)

for model_name in ['resnet18', 'resnet50']:
    for _dataset_type in ['type1']: # ['type2', 'type3', 'type4', 'type5']
        # 计算 loss
        calculate_loss(
            data_csv=pathConvert('../../exp_output/evaluate/{}/{}/evaluate_result.csv'.format(_dataset_type, model_name)),
            outputFile=pathConvert('../../exp_output/analysis/{}/{}'.format(_dataset_type, model_name))
        )

        # predict 和 ground truth 的散点图
        scatterplot_histplot_prediction(
            data_csv=pathConvert('../../exp_output/evaluate/{}/{}/evaluate_result.csv'.format(_dataset_type, model_name)),
            outputFile=pathConvert('../../exp_output/analysis/{}/{}'.format(_dataset_type, model_name))
            )

        # bar height 和 loss 的 heatmap
        barHeight_loss_heatmap(
            data_type=_dataset_type,
            data_csv=pathConvert('../../exp_output/evaluate/{}/{}/evaluate_result.csv'.format(_dataset_type, model_name)),
            bar_height_file=pathConvert('../../exp_output/evaluate/{}/{}/bar_height.npy'.format(_dataset_type, model_name)),
            output_path=pathConvert('../../exp_output/analysis/{}/{}'.format(_dataset_type, model_name))
        )
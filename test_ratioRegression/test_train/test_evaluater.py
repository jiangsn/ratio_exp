'''
@Author: WANG Maonan
@Date: 2021-12-22 23:39:35
@Description: 测试模型, 输出模型预测结果和测试集的数据
@LastEditTime: 2021-12-23 00:56:37
'''
from ratioReg.train.evaluater import predictions_inDataset
from ratioReg.ratioLog.initLog import init_logging
from ratioReg.utils.get_abs_path import getAbsPath

pathConvert = getAbsPath(__file__)
init_logging(log_path=pathConvert('./'), log_level=0)

# 正常情况下测试
for model_name in ['vgg19']:
    for _dataset_type in ['type1']:
        predictions_inDataset(
            model_name=model_name,
            model_path=pathConvert('../../exp_output/model/{}/{}'.format(_dataset_type, model_name)),
            data_path=pathConvert('../../exp_output/dataset/{}'.format(_dataset_type)),
            image_resolution=100,
            patch_size=25,
            output_path=pathConvert('../../exp_output/evaluate/{}/{}'.format(_dataset_type, model_name))
            )

# 对 edit bar 的测试
# predictions_inDataset(
#     model_name='resnet18',
#     model_path=pathConvert('../../exp_output/model/{}'.format('type1')),
#     data_path=pathConvert('../../exp_output/editbar/{}'.format('remove_dots')),
#     image_resolution=100,
#     patch_size=25,
#     output_path=pathConvert('../../exp_output/editbar/{}'.format('remove_dots'))
#     )
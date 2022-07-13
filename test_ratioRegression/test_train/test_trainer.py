'''
@Author: WANG Maonan
@Date: 2021-12-22 23:39:35
@Description: 训练模型
@LastEditTime: 2021-12-23 00:56:37
'''
from ratioReg.train.trainer import train_model
from ratioReg.ratioLog.initLog import init_logging
from ratioReg.utils.get_abs_path import getAbsPath

pathConvert = getAbsPath(__file__)
init_logging(log_path=pathConvert('./'), log_level=0)

for model_name in ['vgg19']:
    for _dataset_type in ['type1']:  # 'type2', 'type3', 'type4', 'type5'
        train_model(
            cuda_index=1,
            model_name=model_name,
            # patch_size=25,
            data_path=pathConvert('../../exp_output/dataset/{}'.format(_dataset_type)),  # 数据集的路径
            lr=0.0001,
            batch_size=256,
            epochs=1,
            output_path=pathConvert(
                '../../exp_output/model/{}/{}/'.format(_dataset_type, model_name)
            ),  # 模型输出的路径
        )


'''
@Author: WANG Maonan
@Date: 2021-12-22 23:39:35
@Description: 生成指定的数据集
@LastEditTime: 2021-12-23 00:56:37
'''
from ratioReg.dataGenerate.GenerateRatioDataset import generate_ratio_dataset
from ratioReg.ratioLog.initLog import init_logging
from ratioReg.utils.get_abs_path import getAbsPath

pathConvert = getAbsPath(__file__)
init_logging(log_path=pathConvert('./'), log_level=0)

for _dataset_type in ['type1', 'type4']:
    data_generate = generate_ratio_dataset(seed=777)
    data_generate.generate_fully_dataset(
        human=False,  # 是否使用 human test
        sampling_method='iid',  # 采样方式
        sampling_reduction=1 / 2,
        dataset_type=_dataset_type,
        sub_dataset_type='move_middle',
        dataset_size={'train': 60000, 'val': 20000, 'test': 20000},  # 每一类的数量
        image_resolution=100,  # 图像的大小, 暂时只支持 100
        output=pathConvert('../../exp_output/dataset/{}'.format(_dataset_type)),
    )


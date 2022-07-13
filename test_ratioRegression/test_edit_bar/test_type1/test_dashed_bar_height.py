'''
@Author: WANG Maonan
@Date: 2021-06-17 01:38:57
@Description: 测试对 type1 将 bar height 改为虚线
@LastEditTime: 2022-01-15 13:49:15
'''
import os
import numpy as np

from ratioReg.utils.save_image import save_ratio_image
from ratioReg.ratioLog.initLog import init_logging
from ratioReg.utils.get_abs_path import getAbsPath
from ratioReg.editBar.type1.dashed_bar_height import change_numpy2dashedBarHeight


pathConvert = getAbsPath(__file__)
init_logging(log_path=pathConvert('./'), log_level=0)

image_numpy_file = pathConvert('../../../exp_output/dataset/type1/test-image.npy')
label_numpy_file = pathConvert('../../../exp_output/dataset/type1/test-label.npy')
bar_height_file = pathConvert('../../../exp_output/dataset/type1/test-bar_height.npy')
dots_positions_numpy_file = pathConvert('../../../exp_output/dataset/type1/test-dots_positions.npy')

# 对原始数据集变化
zip_data = change_numpy2dashedBarHeight(
                    image_numpy_file=image_numpy_file,
                    label_numpy_file=label_numpy_file,
                    bar_height_numpy_file=bar_height_file,
                    dots_positions_numpy_file=dots_positions_numpy_file)
image_numpy_data, label_numpy_data, bar_height_numpy_data, dots_positions_numpy_data = zip_data

# 存储路径
save_path = pathConvert('../../../exp_output/editbar/dashed_bar_height/')
os.makedirs(save_path, exist_ok=True)

# 存储修改后的 numpy 文件
np.save('{}/test-image.npy'.format(save_path), image_numpy_data)
np.save('{}/test-label.npy'.format(save_path), label_numpy_data)
np.save('{}/test-bar_height.npy'.format(save_path), bar_height_numpy_data)
np.save('{}/test-dots_positions.npy'.format(save_path), dots_positions_numpy_data)

# 存储样例图片
image_indexs = [1, 100, 200, 500, 1000]
for image_index in image_indexs:
    convert_image = image_numpy_data[image_index][0] # 修改后的图片
    raw_image = np.load(image_numpy_file)[image_index][0] # 原始的图片

    # 保存图像
    save_ratio_image(convert_image, os.path.join(save_path, '{}_convert_image.jpg'.format(image_index)))
    save_ratio_image(raw_image, os.path.join(save_path, '{}_raw_image.jpg'.format(image_index)))
'''
@Author: WANG Maonan
@Date: 2021-12-22 23:39:35
@Description: 测试 type1-type4 四种的结果
@LastEditTime: 2021-12-23 00:56:37
'''
import numpy as np

from ratioReg.utils.save_image import save_ratio_image
from ratioReg.utils.get_abs_path import getAbsPath
from ratioReg.ratioLog.initLog import init_logging

from ratioReg.ClevelandMcGill.bar_figure_type1 import BarFigure_type1
from ratioReg.ClevelandMcGill.bar_figure_type2 import BarFigure_type2
from ratioReg.ClevelandMcGill.bar_figure_type3 import BarFigure_type3
from ratioReg.ClevelandMcGill.bar_figure_type4 import BarFigure_type4
from ratioReg.ClevelandMcGill.bar_figure_type5 import BarFigure_type5

pathConvert = getAbsPath(__file__)
init_logging(log_path=pathConvert('./'), log_level=0)

# 生成 type1 的图像
for i in range(1, 10):
    data = [8*i, 80] # bar 的组合
    image, bar_height_list, dots_positions_list = BarFigure_type1.data_to_fixed_bottom(data)
    image = image.astype(np.float32)
    save_ratio_image(image, pathConvert('../../exp_output/ClevelandMcGill/type1/{}.jpg'.format(i))) # 保存图像

# 生成 type2 的图像, 此时 black-dots 是在 bar 的中间, 有冗余特征
for i in range(1, 10):
    data = [8*i, 80] # bar 的组合
    image, bar_height_list, dots_positions_list = BarFigure_type2.data_to_fixed_middle(data)
    image = image.astype(np.float32)
    save_ratio_image(image, pathConvert('../../exp_output/ClevelandMcGill/type2/{}.jpg'.format(i))) # 保存图像

# 生成 type3 的图像
for i in range(1, 10):
    data = [8*i, 80] # bar 的组合
    image, bar_height_list, dots_positions_list = BarFigure_type3.data_to_fixed_bottom(data)
    image = image.astype(np.float32)
    save_ratio_image(image, pathConvert('../../exp_output/ClevelandMcGill/type3/{}.jpg'.format(i))) # 保存图像

# 生成 tpye4 的图像
for i in range(1, 10):
    data = [8*i, 80] # bar 的组合
    image, bar_height_list, dots_positions_list = BarFigure_type4.data_to_fixed_middle(data)
    image = image.astype(np.float32)
    save_ratio_image(image, pathConvert('../../exp_output/ClevelandMcGill/type4/{}.jpg'.format(i))) # 保存图像

# 生成 tpye5 的图像
for i in range(1, 10):
    data = [3*i, 30] # bar 的组合
    type5_data = BarFigure_type5.data_to_fixed_middle(data)
    if type5_data != None:
        image, bar_height_list, dots_positions_list = type5_data
    image = image.astype(np.float32)
    save_ratio_image(image, pathConvert('../../exp_output/ClevelandMcGill/type5/{}.jpg'.format(i))) # 保存图像
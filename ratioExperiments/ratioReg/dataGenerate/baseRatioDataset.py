'''
@Author: WANG Maonan
@Date: 2021-12-22 22:36:51
@Description: 数据生成基类
@LastEditTime: 2021-12-23 00:36:14
'''
import logging
from abc import ABC, abstractmethod
from typing import Dict, List

class BaseRatioDataset(ABC):
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
    
    def _generate_ratio_pair(self, human_heights:List) -> Dict[float, List]: 
        """生成每个 ratio 和其 barHeight 的组合:
        1. human_heights -> bar_combination, 例如 [1,2,3] -> [[1,2], [1,3], [2,3]]
        2. bar_combination -> ratio_pair, 例如 [[1,2], [1,3], [2,3]] -> {0.5: [[1,2]], 0.33:[[1,3]], ...}

        Args:
            human_heights (List): 所有 bar height 的值, [10., 12., 15., 18., 22., ...] 等
        """
        bar_combination = [[i, j] for i in human_heights for j in human_heights if i < j]
        ratio_pair = dict() # 比例, 与组成该比例的两个 bar height; 
        for bar_pair in bar_combination:
            bar_pair_ratio = round(min(bar_pair)/max(bar_pair), 2) # 计算两个 bar 的比例
            if bar_pair_ratio not in ratio_pair:
                ratio_pair[bar_pair_ratio] = [bar_pair]
            else:
                ratio_pair[bar_pair_ratio].append(bar_pair)
        return ratio_pair

    def _calc_ratio_pair_cnn(self):
        """不区分 human test. 生成 ratio pair
        """
        self.logger.info('不区分 Human Test.')

        bar_Height = [float(i) for i in range(5, 86)] # 所有可能的 bar height
        ratio_pair = self._generate_ratio_pair(bar_Height)

        return ratio_pair

    def _calc_ratio_pair_human_test(self):
        """区分 human test. 生成 ratio pair
        """
        self.logger.info('区分 Human Test.')

        # #####################
        # 生成 human ratio pair
        # #####################
        human_heights = [10., 12., 15., 18., 22., 26., 32., 38., 46., 56.]
        human_ratio_pair = self._generate_ratio_pair(human_heights)
        human_ratios = [0.18, 0.26, 0.27, 0.38, 0.39, 0.45, 0.46, 0.47, 0.48, \
                        0.55, 0.56, 0.57, 0.58, 0.67, 0.68, 0.69, 0.7, 0.8, \
                        0.81, 0.82, 0.83, 0.84, 0.85]
        for _human_ratio in human_ratio_pair.copy():
            if _human_ratio not in human_ratios:
                del human_ratio_pair[_human_ratio] # 删除不在 human_ratio 中的 ratio
        
        # #####################
        # 生成 other ratio pair
        # #####################
        other_bar_Height = [float(i) for i in range(5, 86)] 
        # 去除 human height 后的 bar height
        for i in human_heights:
            other_bar_Height.remove(i)
        other_ratio_pair = self._generate_ratio_pair(other_bar_Height)
        # 确保 ratio 没有在 human ratio 中
        for _other_ratio in other_ratio_pair.copy():
            if _other_ratio in human_ratios:
                del other_ratio_pair[_other_ratio]

        return (human_ratio_pair, other_ratio_pair)
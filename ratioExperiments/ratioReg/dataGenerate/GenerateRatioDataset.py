'''
@Author: WANG Maonan
@Date: 2021-05-17 20:58:49
@Description: 产生不同的比例的 bar height 的数据集, 并进行保存
@LastEditTime: 2021-12-23 01:10:15
'''
import os
import random

random.seed(777)
import logging
from typing import Dict, List
import numpy as np

from .baseRatioDataset import BaseRatioDataset
from ..utils.save_image import save_ratio_image
from ..ClevelandMcGill.bar_figure_type1 import BarFigure_type1
from ..ClevelandMcGill.bar_figure_type2 import BarFigure_type2
from ..ClevelandMcGill.bar_figure_type3 import BarFigure_type3
from ..ClevelandMcGill.bar_figure_type4 import BarFigure_type4
from ..ClevelandMcGill.bar_figure_type5 import BarFigure_type5


class generate_ratio_dataset(BaseRatioDataset):
    """生成训练, 测试, 验证三部分的数据集, 分别将这三部分的数据集进行保存
    """

    def __init__(self, seed=777) -> None:
        """dataset_type 有五种数据集
        type 1: bar 水平, black-dot 在 2,3; 根据 black-dot 的位置, 还分别是 fixed_bottom, move_bottom, fixed_middle, move_middle
        type 2: bar 堆叠, black-dot 在最下面两个 bar 上;
        type 3: bar 水平, black-dot 在 2, 7 两个 bar 上;
        type 4: bar 堆叠, black-dot 在最上面两个 bar 上;
        type 5: bar 堆叠, black-dot 再 2,3 两个 bar 上;
        """
        self.logger = logging.getLogger(__name__)
        self.rng = np.random.default_rng(seed)
        self.bar_nums = {
            'type1': 10,
            'type2': 8,
            'type3': 10,
            'type4': 8,
            'type5': 8,
        }  # 每一种 type 的 bar 的数量

    def _get_TrainTestVal_ratio(
        self, human: bool = False, sampling_method: str = 'iid', sampling_reduction: float = 1
    ):
        """得到 train_ratio, val_ratio, test_ratio 三者各自的值, 同时返回 all_ratios

        Args:
            human (bool, optional): 是否使用 human test. Defaults to False.
            sampling_reduction (float, optional): 将 train 和 val 变为 sampling_reduction. 
                当取值为 0.5 的时候, train 和 val 随机获得其中的 50%. 
                Defaults to 1.
        """
        if human:  # human test 的数据
            human_ratio_pair, other_ratio_pair = self._calc_ratio_pair_human_test()
            human_ratios = list(human_ratio_pair.keys())  # 得到 human ratio
            others_ratios = list(other_ratio_pair.keys())  # 得到去除 human ratio 剩余的 ratio
            self.rng.shuffle(others_ratios)

            # 训练集, 验证集的样本个数, 测试集就是 human_ratios
            valNum = int(round(len(others_ratios) * 0.25))
            trainNum = len(others_ratios) - valNum
            # 训练集, 验证集, 测试集样本
            test_ratios = human_ratios  # 测试集的 ratio, 例如 [0.12, 0.2, ...]
            val_ratios = others_ratios[:valNum]  # 验证集的 ratio
            train_ratios = others_ratios[valNum:]  # 训练集的 ratio
            # 将 human_ratio_pair, other_ratio_pair 组合为 ratio_pair
            ratio_pair = dict()
            ratio_pair.update(human_ratio_pair)
            ratio_pair.update(other_ratio_pair)
        else:
            ratio_pair = self._calc_ratio_pair_cnn()
            all_ratios = list(ratio_pair.keys())
            self.rng.shuffle(all_ratios)

            # 训练集, 验证集, 测试集样本个数
            testNum = int(round(len(all_ratios) * 0.2))  # 测试集 ratio 的个数
            valNum = int(round(len(all_ratios) * 0.2))
            trainNum = len(all_ratios) - valNum - testNum
            # 训练集, 验证集, 测试集样本
            test_ratios = all_ratios[:testNum]  # 测试集的 ratio, 例如 [0.12, 0.2, ...]
            val_ratios = all_ratios[testNum : testNum + valNum]  # 验证集的 ratio
            train_ratios = all_ratios[testNum + valNum :]  # 训练集的 ratio

        self.logger.info(
            'Sampling Reduction, {} 前, testNum, {}; valNum, {}, trainNum, {}'.format(
                sampling_reduction, len(test_ratios), len(val_ratios), len(train_ratios)
            )
        )
        # 训练集进行 sampling_reduction
        train_num_sampling_reduction = int(len(train_ratios) * sampling_reduction)
        if train_num_sampling_reduction < 3:
            raise ValueError('train_num_sampling_reduction < 3')

        # train_ratios = train_ratios[:train_num_sampling_reduction].copy()
        if sampling_method.lower() == 'iid':
            train_ratios = train_ratios[:train_num_sampling_reduction]
        if sampling_method.lower() == 'ood':
            train_ratios = sorted(train_ratios)[:train_num_sampling_reduction]
        if sampling_method.lower() == 'adv':
            distance = [min([abs(train - test) for test in test_ratios]) for train in train_ratios]
            train_ratios = sorted(
                train_ratios, reverse=True, key=lambda x: distance[train_ratios.index(x)]
            )[:train_num_sampling_reduction]

        if sampling_method.lower() == 'cov':
            train_ratios = sorted(train_ratios)
            selected_ratios = []
            selected_ratios.append(train_ratios.pop(0))
            selected_ratios.append(train_ratios.pop(-1))

            for _ in range(train_num_sampling_reduction - 2):
                distance = [
                    min([abs(train - selected) for selected in selected_ratios])
                    for train in train_ratios
                ]
                train_ratios = sorted(
                    train_ratios, reverse=True, key=lambda x: distance[train_ratios.index(x)]
                )
                selected_ratios.append(train_ratios.pop(0))

            train_ratios = selected_ratios

        self.logger.info(
            'Sampling Reduction, {} 后, testNum, {}; valNum, {}, trainNum, {}'.format(
                sampling_reduction, len(test_ratios), len(val_ratios), len(train_ratios)
            )
        )

        self.logger.debug(
            '-->Train Ratio: {}\n-->Val Ratio: {}\n-->Test Ratio: {}\n'.format(
                sorted(train_ratios), sorted(val_ratios), sorted(test_ratios)
            )
        )  # 打印出每一类中包含的 ratio

        return (train_ratios, val_ratios, test_ratios, ratio_pair)

    def _generate_numpy_dataset(
        self,
        dataset_size: int,
        dataset_type: str,
        sub_dataset_type: str,
        ratio_pair: Dict[float, List],
        ratio_list: List,
        image_resolution: int = 100,
    ):
        """按指定要求生成数据集, 最终返回的数据集为 numpy 格式

        Args:
            dataset_size (int): 数据集的大小, 想要生成的样本数量
            dataset_type (str): 生成的样本种类, 查看 ClevelandMcGill
            sub_dataset_type (str): 每一个 dataset_type 下面会有一些子类
            ratio_pair (Dict[float, List]): 存放每个 ratio 是由哪些 bar combination 组成的, 例如
                {
                    0.83: [[10.0, 12.0], [15.0, 18.0], [38.0, 46.0]],
                    0.67: [[10.0, 15.0], [12.0, 18.0]],
                    0.38: [[10.0, 26.0], [12.0, 32.0]]
                    ...
                }
            ratio_list (List): 要生成的 ratio
            image_resolution (int): 图像的大小. Defaults to 100.
        """
        self.logger.info('开始生成数量大小为 {} 的 numpy dataset.'.format(dataset_size))

        X_train = np.zeros(
            (dataset_size, 1, image_resolution, image_resolution), dtype=np.float32
        )  # Image 数据
        y_train = np.zeros((dataset_size, 1), dtype=np.float32)  # 保存 Label
        bar_height = np.zeros(
            (dataset_size, self.bar_nums[dataset_type]), dtype=np.float32
        )  # 保存一幅图中每个 bar 的高度
        dots_positions = np.zeros((dataset_size, 2, 2), dtype=np.int16)  # 保存 black dots location
        scale = image_resolution / 100.0

        dataset_counter = 0
        while dataset_counter < dataset_size:  # 生成指定数量的数据集
            ratio = random.choice(ratio_list)  # 随机选一个 ratio
            data = random.choice(ratio_pair[ratio])  # 找到对应的 bar height
            label = min(data) / max(data)  # 计算出 label, 实际两个 bar ratio

            try:
                if dataset_type == 'type1':  # 生成 type1 类型的图像
                    if sub_dataset_type == 'fixed_bottom':
                        (
                            image,
                            bar_height_list,
                            dots_positions_list,
                        ) = BarFigure_type1.data_to_fixed_bottom(data, scale)
                    elif sub_dataset_type == 'fixed_middle':
                        (
                            image,
                            bar_height_list,
                            dots_positions_list,
                        ) = BarFigure_type1.data_to_fixed_middle(data)
                    elif sub_dataset_type == 'move_bottom':
                        (
                            image,
                            bar_height_list,
                            dots_positions_list,
                        ) = BarFigure_type1.data_to_move_bottom(data)
                    elif sub_dataset_type == 'move_middle':
                        (
                            image,
                            bar_height_list,
                            dots_positions_list,
                        ) = BarFigure_type1.data_to_move_middle(data)
                elif dataset_type == 'type2':
                    (
                        image,
                        bar_height_list,
                        dots_positions_list,
                    ) = BarFigure_type2.data_to_fixed_middle(data, scale)
                elif dataset_type == 'type3':
                    (
                        image,
                        bar_height_list,
                        dots_positions_list,
                    ) = BarFigure_type3.data_to_fixed_bottom(data, scale)
                elif dataset_type == 'type4':
                    (
                        image,
                        bar_height_list,
                        dots_positions_list,
                    ) = BarFigure_type4.data_to_fixed_middle(data, scale)
                elif dataset_type == 'type5':
                    type5_data = BarFigure_type5.data_to_fixed_middle(data)
                    if type5_data != None:
                        image, bar_height_list, dots_positions_list = type5_data
                    else:
                        continue
                else:
                    self.logger.error('Error, 生成数据集类型不正确, 目前只支持 type1, type2, type3, type4, type5.')
                    return
            except Exception:
                self.logger.error('Error, 图像转换出现问题!')

            image = image.astype(np.float32)

            X_train[dataset_counter][0] = image  # 训练数据集
            y_train[dataset_counter] = label  # label
            bar_height[dataset_counter] = bar_height_list  # 每一个 bar 的高度
            dots_positions[dataset_counter] = dots_positions_list  # 每个 black-dot 的位置
            dataset_counter += 1

        return X_train, y_train, bar_height, dots_positions

    def _generate_fixed_test_data(
        self, save_path: str, dataset_type: str, sub_dataset_type: str, image_resolution: int = 100
    ):
        """产生 ratio 从 0.1-0.9 的数据, 用于测试, 并且保存为图像, 进行可视化

        Args:
            save_path (str): 图像保存的路径
            dataset_type (str): 生成的样本种类, 查看 ClevelandMcGill
            sub_dataset_type (str): 每一个 dataset_type 下面会有一些子类
            image_resolution (int, optional): 图像的大小. Defaults to 100.
        """
        test_data = np.zeros((9, 1, image_resolution, image_resolution), dtype=np.float32)
        test_label = np.zeros((9, 1), dtype=np.float32)
        bar_height = np.zeros(
            (9, self.bar_nums[dataset_type]), dtype=np.float32
        )  # 保存一幅图中每个 bar 的高度
        dots_positions = np.zeros((9, 2, 2), dtype=np.int16)  # 保存 black dots location
        scale = image_resolution / 100.0

        for i in range(1, 10):
            data = [8 * i, 80]  # bar 的组合
            if dataset_type == 'type1':  # 生成 type1 类型的图像
                if sub_dataset_type == 'fixed_bottom':
                    (
                        image,
                        bar_height_list,
                        dots_positions_list,
                    ) = BarFigure_type1.data_to_fixed_bottom(data)
                elif sub_dataset_type == 'fixed_middle':
                    (
                        image,
                        bar_height_list,
                        dots_positions_list,
                    ) = BarFigure_type1.data_to_fixed_middle(data)
                elif sub_dataset_type == 'move_bottom':
                    (
                        image,
                        bar_height_list,
                        dots_positions_list,
                    ) = BarFigure_type1.data_to_move_bottom(data)
                elif sub_dataset_type == 'move_middle':
                    (
                        image,
                        bar_height_list,
                        dots_positions_list,
                    ) = BarFigure_type1.data_to_move_middle(data)
            elif dataset_type == 'type2':
                image, bar_height_list, dots_positions_list = BarFigure_type2.data_to_fixed_middle(
                    data, scale
                )
            elif dataset_type == 'type3':
                image, bar_height_list, dots_positions_list = BarFigure_type3.data_to_fixed_bottom(
                    data, scale
                )
            elif dataset_type == 'type4':
                image, bar_height_list, dots_positions_list = BarFigure_type4.data_to_fixed_middle(
                    data, scale
                )
            elif dataset_type == 'type5':
                data = [i / 8 * 3 for i in data]
                image, bar_height_list, dots_positions_list = BarFigure_type5.data_to_fixed_middle(
                    data, scale
                )
            else:
                self.logger.error('Error, 生成数据集类型不正确.')
                return
            image = image.astype(np.float32)
            image_path = os.path.join(
                save_path,
                'example_{}_{}_{}.jpg'.format(dataset_type, sub_dataset_type, round(i / 10, 2)),
            )
            save_ratio_image(image, image_path, scale)  # 保存图像

            test_data[i - 1][0] = image
            test_label[i - 1] = i / 10  # 返回 ratio
            bar_height[i - 1] = bar_height_list
            dots_positions[i - 1] = dots_positions_list

        return test_data, test_label, bar_height, dots_positions

    def generate_fully_dataset(
        self,
        human: bool = False,
        sampling_method: str = 'iid',
        sampling_reduction: float = 1,
        dataset_type: str = 'type1',
        sub_dataset_type: str = 'fixed_bottom',
        dataset_size: Dict[str, int] = {'train': 60000, 'val': 20000, 'test': 20000},
        image_resolution: int = 100,
        output: str = None,
    ):
        """生成一种类型的 train, val, test 数据集, 和 example 的图像
        """
        os.makedirs(output, exist_ok=True)
        train_ratios, val_ratios, test_ratios, ratio_pair = self._get_TrainTestVal_ratio(
            human, sampling_method, sampling_reduction
        )

        for i, j in {'train': train_ratios, 'val': val_ratios, 'test': test_ratios}.items():
            image, label, bar_height, dots_positions = self._generate_numpy_dataset(
                dataset_size=dataset_size[i],
                dataset_type=dataset_type,
                sub_dataset_type=sub_dataset_type,
                ratio_pair=ratio_pair,
                ratio_list=j,
                image_resolution=image_resolution,
            )
            np.save('{}/{}-{}.npy'.format(output, i, 'image'), image)
            np.save('{}/{}-{}.npy'.format(output, i, 'label'), label)
            np.save('{}/{}-{}.npy'.format(output, i, 'bar_height'), bar_height)
            np.save('{}/{}-{}.npy'.format(output, i, 'dots_positions'), dots_positions)

        self.logger.info('{}-{} 的 train, val, test 三者保存成功.'.format(dataset_type, sub_dataset_type))

        test_data, test_label, bar_height, dots_positions = self._generate_fixed_test_data(
            save_path=output,
            dataset_type=dataset_type,
            sub_dataset_type=sub_dataset_type,
            image_resolution=image_resolution,
        )

        np.save('{}/example-image.npy'.format(output), test_data)  # 保存 example 的 image
        np.save('{}/example-label.npy'.format(output), test_label)  # 保存 example 的 label
        np.save('{}/example-bar_height.npy'.format(output), bar_height)  # 保存 example 的 bar height
        np.save(
            '{}/example-dots_positions.npy'.format(output), dots_positions
        )  # 保存 example 的 dots positions

        self.logger.info('{}-{} 的 9 个测试样本保存完成.\n=======\n'.format(dataset_type, sub_dataset_type))


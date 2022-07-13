'''
@Author: WANG Maonan
@Date: 2021-06-10 11:10:09
@Description: 生成更加有多样性的 bar; 1). 两个 bar 的位置不固定; 2). 两个 bar 不一定是第一个比第二个长
@LastEditTime: 2021-12-23 01:03:54
'''
from typing import Tuple
import numpy as np
np.random.seed(777)
import skimage.draw


class BarFigure_type1:
    
    @staticmethod
    def data_to_fixed_bottom(data, scale=1):
        """fixed black-dot, and black-dot in the bottom
        """
        scale = 1
        data = [data[0], data[1]] # 可能需要从 tuple --> list, 才可以 shuffle
        np.random.shuffle(data) # 随机 bar height
        barchart = np.zeros((100, 100), dtype=np.bool)

        all_values = [0] * 10 # 高度
        all_values[0] = np.random.randint(5, 86)
        all_values[1] = data[0]  # fixed pos 1
        all_values[2] = data[1]  # fixed pos 2
        all_values[3] = np.random.randint(5, 86)
        all_values[4] = np.random.randint(5, 86)
        all_values[5] = np.random.randint(5, 86)
        all_values[6] = np.random.randint(5, 86)
        all_values[7] = np.random.randint(5, 86)
        all_values[8] = np.random.randint(5, 86)
        all_values[9] = np.random.randint(5, 86)

        start = 0
        dots_positions = list() # 记录 black-dots 的位置, 类似 [[96, 14], [96, 23]] 的结果
        for i, d in enumerate(all_values):
            if i == 0:
                start += 2
            elif i == 5:
                start += 8
            else:
                start += 0

            gap = 2
            b_width = 7

            left_bar = start + i*gap + i*b_width # 计算左侧 bar 的位置
            right_bar = start + i*gap + i*b_width + b_width # 计算右侧 bar 的位置


            rr, cc = skimage.draw.line(99, left_bar, 99-int(d)+1, left_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(99, right_bar, 99-int(d)+1, right_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(
                99-int(d)+1, left_bar, 99-int(d)+1, right_bar)
            barchart[rr, cc] = 1

            if i == 1 or i == 2: # 在图上加上黑点
                # place dot here
                barchart[96:97, left_bar+b_width//2:left_bar+b_width//2+1] = 1
                dots_positions.append([96, left_bar+b_width//2]) # 保存黑点的位置

        return barchart, all_values, dots_positions

    @staticmethod
    def data_to_fixed_middle(data):
        """fixed black-dot, and black-dot in the middle of the bar
        """
        data = [data[0], data[1]] # 可能需要从 tuple --> list, 才可以 shuffle
        np.random.shuffle(data) # 随机 bar height
        barchart = np.zeros((100, 100), dtype=np.bool)

        all_values = [0] * 10 # 高度
        all_values[0] = np.random.randint(5, 86)
        all_values[1] = data[0]  # fixed pos 1
        all_values[2] = data[1]  # fixed pos 2
        all_values[3] = np.random.randint(5, 86)
        all_values[4] = np.random.randint(5, 86)
        all_values[5] = np.random.randint(5, 86)
        all_values[6] = np.random.randint(5, 86)
        all_values[7] = np.random.randint(5, 86)
        all_values[8] = np.random.randint(5, 86)
        all_values[9] = np.random.randint(5, 86)

        start = 0
        dots_positions = [] # 记录 black-dots 的位置
        for i, d in enumerate(all_values):
            if i == 0:
                start += 2
            elif i == 5:
                start += 8
            else:
                start += 0

            gap = 2
            b_width = 7

            left_bar = start + i*gap + i*b_width # 计算左侧 bar 的位置
            right_bar = start + i*gap + i*b_width + b_width # 计算右侧 bar 的位置


            rr, cc = skimage.draw.line(99, left_bar, 99-int(d)+1, left_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(99, right_bar, 99-int(d)+1, right_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(
                99-int(d)+1, left_bar, 99-int(d)+1, right_bar)
            barchart[rr, cc] = 1

            if i == 1 or i == 2: # 那两个 bar 需要黑点
                barchart[99-int(d/2), left_bar+b_width//2:left_bar+b_width//2+1] = 1
                dots_positions.append([99-int(d/2), left_bar+b_width//2])

        return barchart, all_values, dots_positions
 
    @staticmethod
    def data_to_move_bottom(data):
        """the black-dot can move and in the bottom
        """
        data = [data[0], data[1]] # 可能需要从 tuple --> list, 才可以 shuffle
        np.random.shuffle(data) # 随机 bar height
        barchart = np.zeros((100, 100), dtype=np.bool)

        data = [data[0], data[1]] # 可能需要从 tuple --> list, 才可以 shuffle
        np.random.shuffle(data) # 随机 bar height
        # 从 0-9 中随机选两个 bar, 来放 black-dot 和之后计算 ratio
        selected_bar = np.random.randint(0, 10, 2)
        while selected_bar[0] == selected_bar[1]: # 判断不能有重复的
            selected_bar = np.random.randint(0, 10, 2)
        
        select_bar_index = 0
        all_values = [0] * 10 # 高度
        for i in range(10):
            if i in selected_bar:
                all_values[i] = data[select_bar_index] # 如果是选定的, 就给出指定高度
                select_bar_index = select_bar_index + 1
            else:
                all_values[i] = np.random.randint(5, 86)

        start = 0
        dots_positions = [] # 记录 black-dots 的位置
        for i, d in enumerate(all_values):
            if i == 0:
                start += 2
            elif i == 5:
                start += 8
            else:
                start += 0

            gap = 2
            b_width = 7

            left_bar = start + i*gap + i*b_width # 计算左侧 bar 的位置
            right_bar = start + i*gap + i*b_width + b_width # 计算右侧 bar 的位置

            rr, cc = skimage.draw.line(99, left_bar, 99-int(d)+1, left_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(99, right_bar, 99-int(d)+1, right_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(
                99-int(d)+1, left_bar, 99-int(d)+1, right_bar)
            barchart[rr, cc] = 1

            if i in selected_bar: # 在指定的两个 bar 上面绘制黑点
                barchart[96:97, left_bar+b_width//2:left_bar+b_width//2+1] = 1
                dots_positions.append([96, left_bar+b_width//2])

        return barchart, all_values, dots_positions

    @staticmethod
    def data_to_move_middle(data):
        """black-dot can move and in the middle of the bar
        """
        data = [data[0], data[1]] # 可能需要从 tuple --> list, 才可以 shuffle
        np.random.shuffle(data) # 随机 bar height
        
        barchart = np.zeros((100, 100), dtype=np.bool)

        # 从 0-9 中随机选两个 bar, 来放 black-dot 和之后计算 ratio
        selected_bar = np.random.randint(0, 10, 2)
        while selected_bar[0] == selected_bar[1]: # 判断不能有重复的
            selected_bar = np.random.randint(0, 10, 2)
        
        select_bar_index = 0
        all_values = [0] * 10 # 高度
        for i in range(10):
            if i in selected_bar:
                all_values[i] = data[select_bar_index] # 如果是选定的, 就给出指定高度
                select_bar_index = select_bar_index + 1
            else:
                all_values[i] = np.random.randint(5, 86)

        start = 0
        dots_positions = [] # 记录 black-dots 的位置
        for i, d in enumerate(all_values): # d 表示 bar height
            if i == 0:
                start += 2
            elif i == 5:
                start += 8
            else:
                start += 0

            gap = 2
            b_width = 7

            left_bar = start + i*gap + i*b_width # 计算左侧 bar 的位置
            right_bar = start + i*gap + i*b_width + b_width # 计算右侧 bar 的位置

            rr, cc = skimage.draw.line(99, left_bar, 99-int(d)+1, left_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(99, right_bar, 99-int(d)+1, right_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(
                99-int(d)+1, left_bar, 99-int(d)+1, right_bar)
            barchart[rr, cc] = 1

            if i in selected_bar: # 在指定的两个 bar 上面绘制黑点
                barchart[99-int(d/2), left_bar+b_width//2:left_bar+b_width//2+1] = 1
                dots_positions.append([99-int(d/2), left_bar+b_width//2])

        return barchart, all_values, dots_positions

    @staticmethod
    def data_to_example():
        """9 个 bar 的高度是固定的
        """
        barchart = np.zeros((100, 100), dtype=np.bool)

        all_values = [0] * 10 # 高度
        for i in range(10):
            all_values[i] = 9*i
        all_values[0] = 1

        start = 0
        for i, d in enumerate(all_values):
            if i == 0:
                start += 2
            elif i == 5:
                start += 8
            else:
                start += 0

            gap = 2
            b_width = 7

            left_bar = start + i*gap + i*b_width # 计算左侧 bar 的位置
            right_bar = start + i*gap + i*b_width + b_width # 计算右侧 bar 的位置


            rr, cc = skimage.draw.line(99, left_bar, 99-int(d)+1, left_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(99, right_bar, 99-int(d)+1, right_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(
                99-int(d)+1, left_bar, 99-int(d)+1, right_bar)
            barchart[rr, cc] = 1

            if i == 1 or i == 2: # 那两个 bar 需要黑点
                barchart[96:97, left_bar+b_width//2:left_bar+b_width//2+1] = 1

        return barchart, all_values

    @staticmethod
    def data_to_custom_bar(height_data, position:Tuple[Tuple]):
        """[summary]

        Args:
            height_data: 每个 bar 的高度
            position (Tuple[Tuple]): black-dots 在哪个 bar 上面, 且在的高度为多少
        """
        barchart = np.zeros((100, 100), dtype=np.bool)

        all_values = [0] * 10 # 高度
        for i in range(10):
            all_values[i] = height_data[i]

        start = 0
        dots_positions = [] # 记录 black-dots 的位置, 类似 [[96, 14], [96, 23]] 的结果
        for i, d in enumerate(all_values):
            if i == 0:
                start += 2
            elif i == 5:
                start += 8
            else:
                start += 0

            gap = 2
            b_width = 7

            left_bar = start + i*gap + i*b_width # 计算左侧 bar 的位置
            right_bar = start + i*gap + i*b_width + b_width # 计算右侧 bar 的位置


            rr, cc = skimage.draw.line(99, left_bar, 99-int(d)+1, left_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(99, right_bar, 99-int(d)+1, right_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(
                99-int(d)+1, left_bar, 99-int(d)+1, right_bar)
            barchart[rr, cc] = 1

            if i == position[0][0]: # 那两个 bar 需要黑点
                # place dot here
                barchart[position[0][1]:position[0][1]+1, left_bar+b_width//2:left_bar+b_width//2+1] = 1
                dots_positions.append([position[0][1], left_bar+b_width//2])
            if i == position[1][0]: # 那两个 bar 需要黑点
                # place dot here
                barchart[position[1][1]:position[1][1]+1, left_bar+b_width//2:left_bar+b_width//2+1] = 1
                dots_positions.append([position[1][1], left_bar+b_width//2])

        return barchart, all_values, dots_positions
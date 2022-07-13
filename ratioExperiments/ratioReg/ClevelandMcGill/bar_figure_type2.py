'''
@Author: WANG Maonan
@Date: 2021-08-06 00:59:34
@Description: 生成第二种类型的 bar
@LastEditTime: 2021-08-10 23:17:09
'''
import numpy as np
np.random.seed(777)
import skimage.draw

class BarFigure_type2:
    @staticmethod
    def data_to_fixed_bottom(data, scale=1):
        """black-dots 固定在下
        """
        data = [data[0], data[1]] # 可能需要从 tuple --> list, 才可以 shuffle
        np.random.shuffle(data) # 随机 bar height
        barchart = np.zeros((100, 100), dtype=np.bool)
        dots_positions = list()

        # we build the barchart to the top
        all_values = [0] * 8
        all_values[0] = data[0]  # fixed pos but max. 56
        current_max = 98-all_values[0]
        # print current_max/4.+1
        all_values[1] = np.random.randint(3, current_max//3.+1)
        all_values[2] = np.random.randint(3, current_max//3.+1)
        all_values[3] = np.random.randint(3, current_max//3.+1)
        current_max = np.sum(all_values[0:4])

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 10, 99-int(current_max), 10)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 40, 99-int(current_max), 40)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values):
            rr, cc = skimage.draw.line(
                99-(int(d)+current), 10, 99-(int(d)+current), 40)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 0: # 第一个黑点
                barchart[96:97, 25:26] = 1
                dots_positions.append([96, 25])

        all_values[4] = data[1]  # fixed pos but max. 56
        current_max = 98-all_values[4]
        # print current_max/4.+1
        all_values[5] = np.random.randint(3, current_max//3.+1)
        all_values[6] = np.random.randint(3, current_max//3.+1)
        all_values[7] = np.random.randint(3, current_max//3.+1)

        current_max = np.sum(all_values[4:])

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 60, 99-int(current_max), 60)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 90, 99-int(current_max), 90)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values[4:]):
            rr, cc = skimage.draw.line(
                99-(int(d)+current), 60, 99-(int(d)+current), 90)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 0: # 第二个黑点
                barchart[96:97, 75:76] = 1
                dots_positions.append([96, 75])

        return barchart, all_values, dots_positions

    @staticmethod
    def data_to_fixed_middle(data, scale=1):
        """black-dots 固定在中部
        """
        data = [data[0], data[1]] # 可能需要从 tuple --> list, 才可以 shuffle
        np.random.shuffle(data) # 随机 bar height
        barchart = np.zeros((100, 100), dtype=np.bool)
        dots_positions = list()

        # we build the barchart to the top
        all_values = [0] * 8
        all_values[0] = data[0]  # fixed pos but max. 56
        current_max = 98-all_values[0]
        # print current_max/4.+1
        all_values[1] = np.random.randint(3, current_max//3.+1)
        all_values[2] = np.random.randint(3, current_max//3.+1)
        all_values[3] = np.random.randint(3, current_max//3.+1)
        current_max = np.sum(all_values[0:4])

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 10, 99-int(current_max), 10)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 40, 99-int(current_max), 40)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values):
            rr, cc = skimage.draw.line(
                99-(int(d)+current), 10, 99-(int(d)+current), 40)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 0: # 第一个黑点
                barchart[99-int(d)//2:99-int(d)//2+1, 25:26] = 1
                dots_positions.append([99-int(d)//2, 25])

        all_values[4] = data[1]  # fixed pos but max. 56
        current_max = 98-all_values[4]
        # print current_max/4.+1
        all_values[5] = np.random.randint(3, current_max//3.+1)
        all_values[6] = np.random.randint(3, current_max//3.+1)
        all_values[7] = np.random.randint(3, current_max//3.+1)

        current_max = np.sum(all_values[4:])

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 60, 99-int(current_max), 60)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 90, 99-int(current_max), 90)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values[4:]):
            rr, cc = skimage.draw.line(
                99-(int(d)+current), 60, 99-(int(d)+current), 90)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 0: # 第二个黑点
                barchart[99-int(d)//2:99-int(d)//2+1, 75:76] = 1
                dots_positions.append([99-int(d)//2, 75])

        return barchart, all_values, dots_positions
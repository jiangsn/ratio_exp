'''
@Author: WANG Maonan
@Date: 2021-08-06 01:00:24
@Description: 生成第三种类型的 bar
@LastEditTime: 2021-08-06 01:00:27
'''
import numpy as np
np.random.seed(777)
import skimage.draw

class BarFigure_type3:
    @staticmethod
    def data_to_fixed_bottom(data, scale=1):
        """black-dots 固定在下, 不会根据 bar height 的不同而变化
        """
        data = [data[0], data[1]] # 可能需要从 tuple --> list, 才可以 shuffle
        np.random.shuffle(data) # 随机 bar height
        barchart = np.zeros((100, 100), dtype=np.bool)
        dots_positions = list()

        # now we need 8 more pairs
        all_values = [0] * 10
        all_values[0] = np.random.randint(5, 86)
        all_values[1] = data[0]  # fixed pos 1
        all_values[2] = np.random.randint(5, 86)
        all_values[3] = np.random.randint(5, 86)
        all_values[4] = np.random.randint(5, 86)
        all_values[5] = np.random.randint(5, 86)
        all_values[6] = data[1]  # fixed pos 2
        all_values[7] = np.random.randint(5, 86)
        all_values[8] = np.random.randint(5, 86)
        all_values[9] = np.random.randint(5, 86)

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

            left_bar = start+i*gap+i*b_width
            right_bar = start+i*gap+i*b_width+b_width

            # print left_bar, right_bar

            rr, cc = skimage.draw.line(99, left_bar, 99-int(d), left_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(99, right_bar, 99-int(d), right_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(
                99-int(d), left_bar, 99-int(d), right_bar)
            barchart[rr, cc] = 1

            if i == 1 or i == 6: # 添加 black-dots
                barchart[96:97, left_bar+b_width//2:left_bar+b_width//2+1] = 1
                dots_positions.append([96, left_bar+b_width//2]) # 保存黑点的位置

        return barchart, all_values, dots_positions
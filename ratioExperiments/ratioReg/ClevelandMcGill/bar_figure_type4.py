'''
@Author: WANG Maonan
@Date: 2021-08-06 01:00:37
@Description: 生成第四种类型的 bar
@LastEditTime: 2021-08-06 01:00:40
'''
import numpy as np
np.random.seed(777)
import skimage.draw

class BarFigure_type4:
    @staticmethod
    def data_to_fixed_middle(data, scale=1):
        """black-dots 固定在下, 不会根据 bar height 的不同而变化
        """
        data = [data[0], data[1]] # 可能需要从 tuple --> list, 才可以 shuffle
        np.random.shuffle(data) # 随机 bar height
        barchart = np.zeros((100, 100), dtype=np.bool)
        dots_positions = list()

        # we build the barchart to the top
        all_values = [0] * 8

        current_max = 98-data[0]

        all_values[0] = np.random.randint(3, current_max//3.+1)
        all_values[1] = np.random.randint(3, current_max//3.+1)
        all_values[2] = np.random.randint(3, current_max//3.+1)

        below_last_sum = np.sum(all_values[0:3])

        all_values[3] = data[0]

        current_max = np.sum(all_values[0:4])

        above_last_sum = current_max

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

            if i == 3: # black-dot
                barchart[99-current+(int(d)//2):99-current+(int(d)//2)+1, 25:26] = 1
                dots_positions.append([99-current+(int(d)//2), 25])

        below_last_sum2 = below_last_sum
        above_last_sum2 = above_last_sum

        # print below_last_sum, above_last_sum

        ctr = 0

        while below_last_sum2 == below_last_sum or above_last_sum2 == above_last_sum:

            if ctr == 20:
                # this happens around 10 times in 100.000 samples.. so not really important
                raise Exception()

            current_max = 98-data[1]

            all_values[4] = np.random.randint(3, current_max//3.+1)
            all_values[5] = np.random.randint(3, current_max//3.+1)
            all_values[6] = np.random.randint(3, current_max//3.+1)

            below_last_sum2 = np.sum(all_values[4:8])

            all_values[7] = data[1]

            current_max = np.sum(all_values[4:])
            above_last_sum2 = current_max

            ctr += 1  # exception counter

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

            if i == 3: # black-dot
                barchart[99-current+(int(d)//2):99-current + (int(d)//2)+1, 75:76] = 1
                dots_positions.append([99-current+(int(d)//2), 75])

        return barchart, all_values, dots_positions
'''
@Author: WANG Maonan
@Date: 2021-07-15 01:44:06
@Description: 将图像保存下来, 不需要保存 bar height
LastEditTime: 2021-12-09 13:03:05
'''
import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def save_ratio_image(image, image_path, scale=1):
    """将 image 保存到 image_path 这个路径下, 

    Args:
        image ([type]): 需要保存的图像
        image_path (str): 保存图像的路径
    """
    os.makedirs(os.path.dirname(image_path), exist_ok=True) # 没有就创建文件夹
    plt.figure(figsize=plt.figaspect(1.0))
    plt.minorticks_off()
    
    # plt.imshow(image, cmap=plt.cm.gray) # 保存为黑白的图像, 只有二值
    plt.imshow(image, cmap=plt.cm.binary) # 保存为黑白的图像, 只有二值
    plt.ylim(int(100*scale)-1, 0) # 设置 y 轴的范围
    plt.xticks([]) # 关闭 x ticks 的显示
    plt.yticks(
        [0, int(20*scale),  int(40*scale),  int(60*scale), int(80*scale),  int(100*scale)-1], # 位置
        [int(100*scale), int(80*scale), int(60*scale), int(40*scale), int(20*scale), 100] # 显示的数值
        )
    
    # 只显示一侧的 ticks
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        direction='out',
        right=False,       # ticks along the right edge are off
        left=True,        # ticks along the left edge are on
        labelbottom=False  # labels along the bottom edge are off
    ) 
    plt.savefig(image_path)
    plt.close()
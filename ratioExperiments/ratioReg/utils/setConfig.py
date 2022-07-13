'''
@Author: WANG Maonan
@Date: 2021-05-17 23:47:38
@Description: 读取, 并使用配置文件
LastEditTime: 2021-07-19 04:18:14
'''
import os
import yaml
import json
from easydict import EasyDict

def setup_config():
    """获取配置信息
    """
    with open(os.path.join('./ratioRegression/entry', 'ratio_regression.yaml'), encoding='utf8') as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)

    # 显示当前的 config 文件
    print(json.dumps(cfg, indent=4))

    return cfg
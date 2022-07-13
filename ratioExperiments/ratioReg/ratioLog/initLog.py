'''
@Author: WANG Maonan
@Date: 2021-12-22 22:06:23
@Description: 配置 root logger
@LastEditTime: 2021-12-22 22:06:24
'''
import os
import logging
import logging.handlers
from datetime import datetime

def init_logging(log_path: str = "logs", log_level: int = 0) -> None:
    """配置 root logger，该 logger 具有 3 种 handler:

    1. sys.stdout << [NOTSET];
    2. <log_path>_debug.log << [DEBUG].
    3. <log_path>_info.log << [INFO];

    Args:
        log_path (str): 日志写入目录 (目录的路径)
        log_level (int): logging 的记录等级, 
            - 0 < DEBUG:10 < INFO:20 < WARNING:30 < ERROR:40 < CRITICAL:50;
            - 低级别模式会记录高级别模式日志

    Returns:
        None
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 创建 log 文件夹
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    # logger formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s [:%(lineno)d] - %(message)s')

    # 创建第一个 handler, 记录所有信息
    now = datetime.strftime(datetime.now(),'%Y-%m-%d_%H_%M_%S_%f')
    ALL_LOG_FILENAME = os.path.join(log_path, '{}.log'.format(now)) # 日志名称

    all_handler = logging.handlers.RotatingFileHandler(
        ALL_LOG_FILENAME, maxBytes=10485760, backupCount=3, encoding='utf-8')
    all_handler.setLevel(logging.DEBUG)
    all_handler.setFormatter(formatter)

    # 创建第二个 handler, 将 INFO 或以上的信息输出到控制台
    info_console_handler = logging.StreamHandler()
    info_console_handler.setLevel(logging.INFO)
    info_console_handler.setFormatter(formatter)

    # 为日志器添加 handler
    root_logger.addHandler(all_handler)
    root_logger.addHandler(info_console_handler)
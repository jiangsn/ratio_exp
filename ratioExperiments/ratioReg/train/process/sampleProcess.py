'''
@Author: WANG Maonan
@Date: 2021-06-08 18:07:24
@Description: 拿出几个具体的例子, 看模型的输出
@LastEditTime: 2021-06-08 18:21:38
'''
import logging

def sample_process(model, test_data_tensor, test_sample_tensor, device):
    """把 test_data 里的预测结果全部输出, 除了宏观指标外, 看一下对于某些值的预测结果
    """
    logger = logging.getLogger(__name__)

    model.eval()  # switch to evaluate mode
    test_data_tensor = test_data_tensor.to(device)
    model_predict = model(test_data_tensor) # 模型的预测结果

    for test_index, (i, j) in enumerate(zip(model_predict, test_sample_tensor)):
        logger.info('==> Sample Index:{}, Model Predict:{:.4f}, Target:{:.4f}.'.format(test_index, i.item(), j.item()))                
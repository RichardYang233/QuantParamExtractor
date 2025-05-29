import torch
from CMSIS_NN_function.arm_fully_connected import _get_input_scale

def flatten_MNIST_image(data):
    '''
    用于将 MINST 的图像数据一维化
    '''
    flattened_data = data.view(data.size(0), -1)

    return flattened_data

def quantize(layer, input) -> torch.Tensor:
    '''
    模拟 self.quant = torch.quantization.QuantStub() 层
    将输入数据量化到 int8
    '''
    scale = _get_input_scale(layer)
    q_input = torch.round(input / scale)

    return q_input.to(torch.int8)
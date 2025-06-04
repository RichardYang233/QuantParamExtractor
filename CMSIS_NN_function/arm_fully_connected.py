import torch
import torch.nn.functional as F
from .utils import *

# --------------------------------------------------------------------------------
# 主要功能
# --------------------------------------------------------------------------------

class FCQuantSim:

    def __init__(self, last_layer, current_layer):

        # 层信息
        self.last_layer = last_layer
        self.current_layer = current_layer

        # 输入 & 输出
        self.input = None
        self.output = None

        # 权重 & 偏置
        self.weight_int8 = get_weight(current_layer)
        self.bias_int32 = calculate_bias_int32(current_layer, last_layer)
        
        # 反量化参数
        self.multiplier, self.shift = calcu_multi_and_shift_with_scale(last_layer, current_layer)
        self.zero_point = get_output_zero_point(current_layer)

    def arm_fully_connected_s8(self, input: torch.tensor):
        '''
        模拟全连接层的量化推理,
        等价 CMSIS-NN @arm_fully_connected_s8 的计算方式

        Arg:
            input (torch.Tensor): 输入值 [int8]
        Returns:
            output_int8_relu (torch.Tensor): 输出值 [int8]
        Note: 
            计算流程：
            - input @ weight.T + bias
            - dequant [int32] to [int8]
            - relu
        '''
        input_int32 = input.to(torch.int32)
        weight_int32 = self.weight_int8.to(torch.int32)
        bias_int32 = self.bias_int32.to(torch.int32)

        # output = input @ weight.T + bias
        output_int32 = linear(input_int32, weight_int32, bias_int32)
        # dequant with [mult] and [shift]
        output_int8 = dequant_with_mult_and_shift(output_int32, self.multiplier, self.shift, self.zero_point)
        # relu
        output_int8_relu = torch.relu(output_int8) # output_int8_relu = torch.maximum(output_int8, torch.tensor(current_layer.zero_point, dtype=output_int8.dtype))

        return output_int8_relu

def torch_fully_connected_s8(input, last_layer, current_layer):
    '''
    模拟全连接层的量化推理,
    目前看来可以等价于 pytorch 的量化推理    # NOTE: pytorch的实际量化流程待查找
    '''
    input_int32  = input.to(torch.int32)
    weight_int32 = get_weight(current_layer).to(torch.int32)
    bias_int32   = calculate_bias_int32(current_layer, last_layer).to(torch.int32) 
    
    # output = input @ weight.T + bias
    output_int32 = linear(input_int32, weight_int32, bias_int32)

    # dequant with [scale] and [zero_point]
    output_int8 = dequant_with_scale_and_zero_point(output_int32, last_layer, current_layer)

    # relu
    output_int8_relu = torch.relu(output_int8)

    return output_int8_relu


import torch
import torch.nn.functional as F
from .utils import *


class ConvQuantSim:
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

    def arm_convolve_s8(self, input: torch.tensor):
        '''
        模拟卷积层的量化推理
        '''
        input_int32 = input.to(torch.int32)
        weight_int32 = self.weight_int8.to(torch.int32)
        bias_int32 = self.bias_int32.to(torch.int32)

        # NOTE: 这里的计算顺序不对会报错
        # conv2d
        output_int32 = F.conv2d(input_int32, weight_int32, bias_int32) # , stride=1, padding=1
        # pool
        output_int32_pool = F.max_pool2d(output_int32, kernel_size = 2, stride = 2)
        # dequant with [mult] and [shift]
        output_int8_pool = dequant_with_mult_and_shift(output_int32_pool, self.multiplier, self.shift, self.zero_point)
        # relu
        output_int8_pool_relu = torch.relu(output_int8_pool)

        return output_int8_pool_relu



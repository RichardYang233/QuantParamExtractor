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

        # conv2d
        output_int32 = F.conv2d(input_int32, weight_int32, bias_int32) # stride=1, padding=0
        # relu
        output_int32_relu = torch.relu(output_int32)
        # dequant with [mult] and [shift]
        output_int8_relu = dequant_with_mult_and_shift(output_int32_relu, self.multiplier, self.shift, self.zero_point)
        output_int8_relu = output_int8_relu.to(torch.int32) # 如果是 int8 会导致 pool 步骤报错
        # pool
        output_int8_relu_pool = F.max_pool2d(output_int8_relu, kernel_size = 2, stride = 2)

        return output_int8_relu_pool.to(torch.int8) # 再转回 int8
    
    def _get_multiplier(self):
        return self.multiplier

    def _get_shift(self):
        return self.shift

    def _get_zero_point(self):
        return self.zero_point
    
    # 打印 CMSIS-NN 反量化所需参数
    def print_multi_shift_zero_point(self):
        dict = {'multiplier': self._get_multiplier(),
                'shift': self._get_shift(),
                'zero_point': self._get_zero_point()}
        print(dict)
        


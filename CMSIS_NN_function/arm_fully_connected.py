import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.weight_int8 = _get_weight(current_layer)
        self.bias_int32 = calculate_bias_int32(current_layer, last_layer)
        
        # 反量化参数
        self.multiplier, self.shift = calcu_multi_and_shift_with_scale(last_layer, current_layer)
        self.zero_point = _get_output_zero_point(current_layer)

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
    weight_int32 = _get_weight(current_layer).to(torch.int32)
    bias_int32   = calculate_bias_int32(current_layer, last_layer).to(torch.int32) 
    
    # output = input @ weight.T + bias
    output_int32 = linear(input_int32, weight_int32, bias_int32)

    # dequant with [scale] and [zero_point]
    output_int8 = dequant_with_scale_and_zero_point(output_int32, last_layer, current_layer)

    # relu
    output_int8_relu = torch.relu(output_int8)

    return output_int8_relu

def torch_conv2d_s8(input, last_layer, current_layer):
    '''
    模拟卷积层的量化推理
    '''
    input_int32 = input.to(torch.int32)
    weight_int32 = _get_weight(current_layer).to(torch.int32)
    bias_int32 = calculate_bias_int32(current_layer, last_layer).to(torch.int32)

    # NOTE: 这里的计算顺序不对会报错
    # conv2d
    output_int32 = F.conv2d(input_int32, weight_int32, bias_int32) # , stride=1, padding=1
    # pool
    output_int32_pool = F.max_pool2d(output_int32, kernel_size = 2, stride = 2)
    # dequant with [scale] and [zero_point] 
    output_int8_pool = dequant_with_scale_and_zero_point(output_int32_pool, last_layer, current_layer) # NOTE： 暂时使用，需更换到 @dequant_with_mult_and_shift()
    # relu
    output_int8_pool_relu = torch.relu(output_int8_pool)
    
    return output_int8_pool_relu


# --------------------------------------------------------------------------------
# 
# --------------------------------------------------------------------------------

def linear(input, weight, bias):
    output = input @ weight.T + bias
    return output.to(torch.int32)

def calculate_bias_int32(current_layer, last_layer):
    '''
    NOTE: 暂用这种计算方式，不一定合理
    '''
    bias_float64 = _get_bias(current_layer)
    weight_scale = _get_weight_scale(current_layer)
    input_scale = _get_input_scale(last_layer)

    bias_int32 = torch.round(bias_float64 / (weight_scale * input_scale))
    return bias_int32.to(torch.int32)

def dequant_with_scale_and_zero_point(input_int32, last_layer, current_layer):

    input_scale  = _get_input_scale(last_layer)
    weight_scale =  _get_weight_scale(current_layer)
    output_scale = _get_output_scale(current_layer)
    output_zero_point = _get_output_zero_point(current_layer)

    scale = input_scale * weight_scale / output_scale

    output_int8 = (input_int32 * scale) + output_zero_point
    output_int8 = (torch.round(output_int8)).clamp(-128, 127)

    return output_int8.to(torch.int8)

def dequant_with_mult_and_shift(data_int32, multiplier, shift, zero_point):

    # 计算方法1 参考自:
    # CMSIS-NN库 @arm_nn_requantize() 中的 CMSIS_NN_USE_SINGLE_ROUNDING 分支
    total_shift = 31 - shift
    data_int32 = data_int32.to(torch.int64) * multiplier
    data_int32 = (data_int32 + (1 << (total_shift - 1))) >> total_shift
    data_int32 += zero_point
    return data_int32.clamp(-128, 127).to(torch.int8)

    # # 计算方法2 参考自:
    # CMSIS-NN库 @arm_nn_requantize() 中的 默认 分支
    def LEFT_SHIFT(s):
        return s if s > 0 else 0

    def RIGHT_SHIFT(s):
        return 0 if s > 0 else -s

    # Step 1: Apply left shift if shift > 0
    val = data_int32 * (1 << LEFT_SHIFT(shift))

    # Step 2: Doubling high multiply (simulate 64-bit * 64-bit >> 31 with rounding)
    # Compute int64(val) * int64(multiplier) + (1 << 30), then >> 31
    val = val.to(torch.int64)
    mult = torch.tensor(multiplier, dtype=torch.int64)
    rounding_offset = 1 << 30
    prod = val * mult + rounding_offset
    doubled_high = (prod >> 31).to(torch.int32)

    # Step 3: Divide by power of two with rounding
    exponent = RIGHT_SHIFT(shift)
    if exponent != 0:
        remainder_mask = (1 << exponent) - 1
        remainder = doubled_high & remainder_mask
        result = doubled_high >> exponent

        threshold = remainder_mask >> 1
        result += ((doubled_high < 0) & (remainder > threshold)).int()
        result += ((doubled_high >= 0) & (remainder > threshold)).int()
    else:
        result = doubled_high

    # Step 4: Add zero_point and clamp to int8
    result = result + zero_point
    result = result.clamp(-128, 127)

    return result.to(torch.int8)

def calcu_multi_and_shift_with_scale(last_layer, current_layer):

    input_scale  = _get_input_scale(last_layer)
    weight_scale = _get_weight_scale(current_layer)
    output_scale = _get_output_scale(current_layer)

    real_scale = float((input_scale * weight_scale) / output_scale)

    # NOTE: 凭感觉写的，不确定是否能获取最佳值
    def calcu_shift_multi(scale):
        shift = 0
        multiplier = 0
        while True: 
            shift += 1;
            multiplier = round(scale * (1 << shift))
            if multiplier < ((1 << 30) - 1): # 不超出 int32
                continue
            else:
                return multiplier, shift

    multiplier, shift = calcu_shift_multi(real_scale)
    shift = 31 - shift  # 此 shift 值对应 CMSIS-NN 中的 shift 值

    return multiplier, shift

# --------------------------------------------------------------------------------
# 基础功能
# --------------------------------------------------------------------------------

def _get_bias(current_layer) -> float:
    return current_layer.bias()

def _get_weight(current_layer):
    # torch.int_repr 以获取量化后的值
    return torch.int_repr(current_layer.weight())

def _get_weight_scale(current_layer):
    return current_layer.weight().q_scale()

def _get_input_scale(last_layer):
    return last_layer.scale

def _get_output_scale(current_layer):
    return current_layer.scale

def _get_output_zero_point(current_layer):
    return current_layer.zero_point
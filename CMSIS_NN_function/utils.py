import torch

# TODO: 这个函数名需更改
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
    scale = get_input_scale(layer)
    q_input = torch.round(input / scale)

    return q_input.to(torch.int8)

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
    bias_float64 = get_bias(current_layer)
    weight_scale = get_weight_scale(current_layer)
    input_scale = get_input_scale(last_layer)

    bias_int32 = torch.round(bias_float64 / (weight_scale * input_scale))
    return bias_int32.to(torch.int32)

def dequant_with_scale_and_zero_point(input_int32, last_layer, current_layer):

    input_scale  = get_input_scale(last_layer)
    weight_scale =  get_weight_scale(current_layer)
    output_scale = get_output_scale(current_layer)
    output_zero_point = get_output_zero_point(current_layer)

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

    input_scale  = get_input_scale(last_layer)
    weight_scale = get_weight_scale(current_layer)
    output_scale = get_output_scale(current_layer)

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

def get_bias(current_layer) -> float:
    return current_layer.bias()

def get_weight(current_layer):
    # torch.int_repr 以获取量化后的值
    return torch.int_repr(current_layer.weight())

def get_weight_scale(current_layer):
    return current_layer.weight().q_scale()

def get_input_scale(last_layer):
    return last_layer.scale

def get_output_scale(current_layer):
    return current_layer.scale

def get_output_zero_point(current_layer):
    return current_layer.zero_point
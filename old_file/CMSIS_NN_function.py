import torch


FLAG = []

# ----- 主要功能函数 ----- #

# NOTE: 弯路
def arm_fully_connected_s8_TEST(last_layer, current_layer, input,
                                multiplier, shift):
    '''
    一个看起来精度还可以的方法，没有理论依据
    multiplier, shift 来自于 test1.py 中生成值
    '''
    input_int32  = input.to(torch.int32)
    weight_int32 = _get_weight(current_layer).to(torch.int32)
    bias_int32   = calculate_bias_int32(current_layer, last_layer).to(torch.int32) # NOTE: 暂用这种计算方式，不一定合理

    # output = input @ weight.T + bias
    output_int32 = linear(input_int32, weight_int32, bias_int32)

    zero_point = _get_output_zero_point(current_layer)
    output_int8 = ((output_int32.long() * multiplier) >> (31 + shift)) + zero_point
    output_int8 = output_int8.clamp(-128, 127).to(torch.int8)
    
    # relu
    output_int8_relu = torch.relu(output_int8) # output_int8_relu = torch.maximum(output_int8, torch.tensor(current_layer.zero_point, dtype=output_int8.dtype))

    return output_int8_relu

# NOTE: 弯路
def get_output_int32(current_layer, last_layer, input, 
                                    Multiplier = None, 
                                    Shift = None):
    input_int32 = input.to(torch.int32)
    weight_int32 = _get_weight(current_layer).to(torch.int32)
    bias_int32 = calculate_bias_int32(current_layer, last_layer).to(torch.int32) 

    output_int32 = linear(input_int32, weight_int32, bias_int32)

    FLAG.append(output_int32)

# ----- 所需函数 ----- #

# NOTE: 弯路
def calcu_multi_and_shift_with_input_range(data_int32):

    max_raw_data = data_int32.max().item()
    min_raw_data = data_int32.min().item()
    max_int8 = torch.iinfo(torch.int8).max
    min_int8 = torch.iinfo(torch.int8).min

    scale = (max_int8 - min_int8) / (max_raw_data - min_raw_data)

    # 计算 multiplier 和 shift
    real_multiplier = scale
    shift = 0
    while real_multiplier < 0.5:
        real_multiplier *= 2
        shift += 1
    multiplier = int(round(real_multiplier * (1 << 31)))

    q_data = ((data_int32.long() * multiplier) >> (31 + shift)).clamp(-128, 127)

    return multiplier, shift, q_data







    






    








# def linear():

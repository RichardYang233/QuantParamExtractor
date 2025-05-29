import torch
import torch.nn as nn
import torch.nn.functional as F

# 非对称量化

def linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype):
      
    scaled_and_shifted_tensor = tensor / scale + zero_point

    rounded_tensor = torch.round(scaled_and_shifted_tensor)

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)
    return q_tensor

def linear_dequantization(quantized_tensor, scale, zero_point):
    return scale * (quantized_tensor.float() - zero_point)

def get_q_scale_and_zero_point(tensor, dtype):
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    r_min = tensor.min().item()
    r_max = tensor.max().item()

    scale = (r_max - r_min) / (q_max - q_min)
    zero_point = q_min - (r_min / scale)

    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else:
        zero_point = int(round(zero_point))

    return scale, zero_point

def linear_quantization(tensor, dtype):
    scale, zero_point = get_q_scale_and_zero_point(tensor, dtype=dtype)
    quantized_tensor =  linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype=dtype)

    return quantized_tensor, scale, zero_point

# 对称量化

def get_q_scale_symmetric(tensor, dtype):
    r_max = tensor.abs().max().item()
    q_max = torch.iinfo(dtype).max

    return r_max/q_max

def linear_quantization_symmetric(tensor, dtype):
    scale = get_q_scale_symmetric(tensor, dtype)
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale=scale, zero_point=0, dtype=dtype)

    return quantized_tensor, scale

        
# 逐通道量化

def linear_q_symmetric_per_channel(tensor, dim, dtype=torch.int8):
    # dim = 0 按行量化
    output_dim = tensor.shape[dim] 
    # store the scales
    scale = torch.zeros(output_dim)

    for index in range(output_dim):
        sub_tensor = tensor.select(dim, index)
        scale[index] = get_q_scale_symmetric(sub_tensor, dtype=dtype)

    # reshape the scale
    scale_shape = [1] * tensor.dim()
    scale_shape[dim] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale=scale, zero_point=0, dtype=dtype)

    return quantized_tensor, scale








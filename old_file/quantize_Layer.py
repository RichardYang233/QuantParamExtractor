import torch
import torch.nn as nn
import torch.nn.functional as F

from linear_quantize import *


class Reshape(nn.Module):
    def forward(self, x):
        # current version PyTorch does not support integer-based MaxPool
        return x.view(x.shape[0], -1)
    
class Relu(nn.Module):
    def forward(self, input):
        return nn.ReLU(input.to(torch.int8))
    
class QuantizedLinear(nn.Module):
    def __init__(self, weight, bias,
                 r_input, r_weight, r_bias,
                 input_scale, input_zero_point,
                 output_scale, output_zero_point):
        super().__init__()

        self.register_buffer('r_input', r_input)
        self.register_buffer('r_weight', r_weight)
        self.register_buffer('r_bias', r_bias)
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)
        self.input_scale =  input_scale
        self.input_zero_point = input_zero_point
        self.output_scale = output_scale
        self.output_zero_point = output_zero_point

    def forward(self, input):

        # TODO: input_offset, multiplier, shift, output_offset 值的确定

        q_input = linear_q_with_scale_and_zero_point(input, self.input_scale, self.input_zero_point, dtype=torch.int8)

        output = torch.nn.functional.linear(q_input.to(torch.int32), self.weight.to(torch.int32), self.bias.to(torch.int32))

        q_output = linear_q_with_scale_and_zero_point(output, self.output_scale, self.output_zero_point, dtype=torch.int8)
        


        return q_output


if __name__ == "__main__":

    x = torch.tensor([[2,1,1,1],
                      [1,1,1,1]])       

    quantized_backbone = [] # 新的模型容器
    quantized_backbone.append(Reshape())
    quantized_backbone.append(QuantizedLinear(x,x,x,x,x,x))
    quantized_backbone.append(QuantizedLinear(x,x,x,x,x,x))
    quantized_model = nn.Sequential(*quantized_backbone)
    
    print(quantized_model)
    result = quantized_model(x)
    print(result)
import torch
import torch.nn as nn
import torch.nn.functional as F

from linear_quantize import *


def w8_a16_forward(weight, input, scales, bias=None):
    casted_weight = weight.to(input.dtype)
    output = F.linear(input, casted_weight) * scales
    if bias is not None:
        output = output + bias
    return output

class W8A16LinearLayer(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True, dtype=torch.float32):
        super().__init__()

        self.register_buffer(
            "int8_weights", torch.randint(-128, 127, (out_feature, in_feature), dtype=torch.int8))
        self.register_buffer(
            "scales", torch.randn((out_feature), dtype=dtype))
        if bias:
            self.register_buffer(
                "bias", torch.randn((1, out_feature), dtype=dtype))
        else:
            self.bias = None

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)
        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights / scales.unsqueeze(1)).to(torch.int8)

        self.int8_weights = int8_weights
        self.scales = scales

    def forward(self, input):
        return w8_a16_forward(self.int8_weights, input, self.scales, self.bias)
    


# module = W8A16LinearLayer(4, 8)
# print(module.int8_weights)

# random_matrix = torch.randn((4, 8), dtype=torch.bfloat16)
# module.quantize(random_matrix)

# print(module.int8_weights)


def replace_linear_with_target_and_quantize(module: nn.Module, target_class, 
                               module_name_to_exclude):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias
            old_weight = child.weight

            new_module =  target_class(child.in_features,
                                       child.out_features, old_bias is not None,
                                       child.weight.dtype)

            setattr(module, name, new_module)

            getattr(module, name).quantize(old_weight)

            if old_bias is not None:
                getattr(module, name).bias = old_bias
        else:
            replace_linear_with_target_and_quantize(child, target_class, module_name_to_exclude)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1, 1)
        self.linear_1 = nn.Linear(1, 1)
        self.linear_2 = nn.Linear(1, 1, bias=False)
        self.lm_head = nn.Linear(1, 1, bias=False)

model_1 = Net()
model_2 = Net()

replace_linear_with_target_and_quantize(model_1, W8A16LinearLayer, ["lm_head"])
print(model_1)
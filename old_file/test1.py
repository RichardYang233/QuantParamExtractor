import torch

from models import *
from train import *
from eval import *
from dataset import *
from utils import *


model_util = generate_q_model(QuantizedFCNet(), FCNET_PARAMS_SAVE_PATH)

_, test_loader = get_DataLoader(64)


Multiplier = []
Shift = []

correct = 0
total = 0

for batch in tqdm(test_loader):

    inputs, labels = batch 
    
    # quant
    quant_layer_input = flatten_MNIST_image(inputs)
    quant_layer_output = quantize(model_util.quant, quant_layer_input)

    # hidden_layer
    hidden_layer_input = quant_layer_output
    get_output_int32(model_util.hidden_layer, model_util.quant, hidden_layer_input)

flattened = [t.flatten() for t in FLAG]  # 都变成一维
combined = torch.cat(flattened)          # 可以拼接
multiplier, shift, data_int8 = calcu_multi_and_shift_with_input_range(combined)
FLAG.clear()

print(data_int8.max().item())
print(data_int8.min().item())
print(f"multiplier = {multiplier}")
print(f"shift = {shift}")

for batch in tqdm(test_loader):

    inputs, labels = batch 
    
    # quant
    quant_layer_input = flatten_MNIST_image(inputs)
    quant_layer_output = quantize(model_util.quant, quant_layer_input)

    # hidden_layer
    hidden_layer_input = quant_layer_output
    hidden_layer_output = arm_fully_connected_s8_TEST(model_util.quant       ,
                                                      model_util.hidden_layer, 
                                                      hidden_layer_input          ,
                                                      multiplier, shift)

    # output_layer
    output_layer_input = hidden_layer_output
    get_output_int32(model_util.output_layer, model_util.hidden_layer, output_layer_input) 
    
flattened = [t.flatten() for t in FLAG]  # 都变成一维
combined = torch.cat(flattened)          # 可以拼接
multiplier, shift, data_int8 = calcu_multi_and_shift_with_input_range(combined)
FLAG.clear()

print(data_int8.max().item())
print(data_int8.min().item())
print(f"multiplier = {multiplier}")
print(f"shift = {shift}")

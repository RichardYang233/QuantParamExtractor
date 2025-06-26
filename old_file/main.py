import torch

from models import *
from train import *
from eval import *
from dataset import *
from utils import *


model_util = generate_q_model(QuantizedFCNet(), FCNET_PARAMS_SAVE_PATH)

Multiplier = []
Shift = []

# ------------ 量化模型推理 ------------ #

_, test_loader = get_DataLoader(64)
correct = 0
total = 0

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
                                                      multiplier = 1185513671, shift = 11)

    # output_layer
    output_layer_input = hidden_layer_output
    output_layer_output = arm_fully_connected_s8_TEST(model_util.hidden_layer, 
                                                      model_util.output_layer,
                                                      output_layer_input          ,
                                                      multiplier = 1110397882, shift = 9)
    
    # Argmax   
    results = torch.argmax(output_layer_output, dim=1)
    count = (results == labels).sum().item()

    correct += count
    total += results.shape[0]

print(f"CMSIS-NN计算精度: {correct/total*100:.2f}%")

accuracy = evaluate(model_util, test_loader)
print(f"量化模型精度: {accuracy:.2f}%")

model = load_state_dict_2_model(FCNet(), FCNET_PARAMS_SAVE_PATH) 
accuracy = evaluate(model, test_loader)
print(f"原始模型精度: {accuracy:.2f}%")
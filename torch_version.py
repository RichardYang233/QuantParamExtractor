import torch

from models import *
from train import *
from eval import *
from dataset import *
from utils import *
from CMSIS_NN_function import *


quantized_model = generate_q_model(QuantizedFCNet(), FCNET_PARAMS_SAVE_PATH)

_, test_loader = get_DataLoader(64)

correct = 0
total = 0

# ---------- 推理 ---------- #
for batch in tqdm(test_loader):

    inputs, labels = batch 
    
    # quant
    quant_layer_input = flatten_MNIST_image(inputs)
    quant_layer_output = quantize(quantized_model.quant, quant_layer_input)

    # hidden_layer
    hidden_layer_input = quant_layer_output
    hidden_layer_output = torch_fully_connected_s8(hidden_layer_input           ,
                                                   quantized_model.quant        ,
                                                   quantized_model.hidden_layer )
    # output_layer
    output_layer_input = hidden_layer_output
    output_layer_output = torch_fully_connected_s8(output_layer_input           ,
                                                   quantized_model.hidden_layer ,
                                                   quantized_model.output_layer )
     
    results = torch.argmax(output_layer_output, dim=1)

    count = (results == labels).sum().item()
    correct += count
    total += results.shape[0]

print(f"模拟torch量化模型精度: {correct/total*100:.2f}%")

accuracy = evaluate(quantized_model, test_loader)
print(f"torch量化模型精度: {accuracy:.2f}%")

model = load_state_dict_2_model(FCNet(), FCNET_PARAMS_SAVE_PATH) 
accuracy = evaluate(model, test_loader)
print(f"原始模型精度: {accuracy:.2f}%")









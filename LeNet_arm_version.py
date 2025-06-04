import torch
import torch.nn as nn

from models import *
from train import *
from eval import *
from dataset import *
from utils import *
from CMSIS_NN_function import *


quantized_model = generate_q_LeNet(QuantizedLeNet(), LENET_PARAMS_SAVE_PATH)

_, test_loader = get_DataLoader(64)

correct = 0
total = 0

# ---------- 推理 ---------- #

conv1_simulator = ConvQuantSim(quantized_model.quant, quantized_model.conv1)
conv2_simulator = ConvQuantSim(quantized_model.conv1, quantized_model.conv2)
fc1_simulator = FCQuantSim(quantized_model.conv2, quantized_model.fc1)
fc2_simulator = FCQuantSim(quantized_model.fc1, quantized_model.fc2)
fc3_simulator = FCQuantSim(quantized_model.fc2, quantized_model.fc3)

for batch in tqdm(test_loader):

    inputs, labels = batch

    # quant
    quant_layer_output = quantize(layer = quantized_model.quant, input = inputs)

    # conv1 & relu & maxpool1
    conv1_input = quant_layer_output
    conv1_output = conv1_simulator.arm_convolve_s8(conv1_input)

    # conv2 & relu & maxpool2
    conv2_input = conv1_output
    conv2_output = conv2_simulator.arm_convolve_s8(conv2_input)

    # flatten
    conv2_output = conv2_output.reshape(conv2_output.shape[0], -1)

    # fc1
    fc1_input = conv2_output
    fc1_output = fc1_simulator.arm_fully_connected_s8(fc1_input)

    # fc2
    fc2_input = fc1_output
    fc2_output = fc2_simulator.arm_fully_connected_s8(fc2_input)

    # fc3
    fc3_input = fc2_output
    fc3_output = fc3_simulator.arm_fully_connected_s8(fc3_input)

    # argmax
    results = torch.argmax(fc3_output, dim=1)

    # 
    count = (results == labels).sum().item()
    correct += count
    total += results.shape[0]


# ---------- 评估 ---------- #

print(f"模拟CMSIS-NN计算精度: {correct/total*100:.2f}%")

accuracy = evaluate(quantized_model, test_loader)
print(f"torch量化模型精度: {accuracy:.2f}%")

model = load_state_dict_2_model(LeNet(), LENET_PARAMS_SAVE_PATH) 
accuracy = evaluate(model, test_loader)
print(f"原始模型精度: {accuracy:.2f}%")
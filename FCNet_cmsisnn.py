import torch

from models import *
from train import *
from eval import *
from utils import *
from CMSIS_NN_simulator import *


quantized_model = generate_q_model(QuantizedFCNet(), FCNET_PARAMS_SAVE_PATH)

_, test_loader = get_DataLoader(64)

correct = 0
total = 0

# ---------- 推理 ---------- #

hidden_layer_simulator = FCQuantSim(quantized_model.quant, quantized_model.hidden_layer)
output_layer_simulator = FCQuantSim(quantized_model.hidden_layer, quantized_model.output_layer)

for batch in tqdm(test_loader):

    inputs, labels = batch
    
    # flatten & quant
    quant_layer_input = flatten_MNIST_image(inputs)
    quant_layer_output = quantize(quantized_model.quant, quant_layer_input)

    # hidden_layer
    hidden_layer_input = quant_layer_output
    hidden_layer_output = hidden_layer_simulator.arm_fully_connected_s8(hidden_layer_input)

    # output_layer
    output_layer_input = hidden_layer_output
    output_layer_output = output_layer_simulator.arm_fully_connected_s8(output_layer_input)

    # argmax
    results = torch.argmax(output_layer_output, dim=1)

    # 
    count = (results == labels).sum().item()
    correct += count
    total += results.shape[0]

# ---------- 评估 ---------- #

print(f"模拟CMSIS-NN计算精度: {correct/total*100:.2f}%")

accuracy = evaluate(quantized_model, test_loader)
print(f"torch量化模型精度: {accuracy:.2f}%")

model = load_state_dict_2_model(FCNet(), FCNET_PARAMS_SAVE_PATH) 
accuracy = evaluate(model, test_loader)
print(f"原始模型精度: {accuracy:.2f}%")

# ---------- CMSIS-NN 推理所需参数导出 ---------- #

# # 保存 weight、bias 至 C数组
# print(hidden_layer_simulator.weight_int8.size())
# export_tensor_as_c_array(hidden_layer_simulator.weight_int8, "hidden_layer_weight", "hidden_layer_weight", "int8_t")
# print(hidden_layer_simulator.bias_int32.size())
# export_tensor_as_c_array(hidden_layer_simulator.bias_int32, "hidden_layer_bias", "hidden_layer_bias", "int32_t")
# print(output_layer_simulator.weight_int8.size())
# export_tensor_as_c_array(output_layer_simulator.weight_int8, "output_layer_weight", "output_layer_weight", "int8_t")
# print(output_layer_simulator.bias_int32.size())
# export_tensor_as_c_array(output_layer_simulator.bias_int32, "output_layer_bias", "output_layer_bias", "int32_t")

# print 反量化参数 #

hidden_layer_simulator.print_multi_shift_zero_point()
output_layer_simulator.print_multi_shift_zero_point()






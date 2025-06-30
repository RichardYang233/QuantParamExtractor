import torch

from models import *
from train import *
from eval import *
from utils import *
from CMSIS_NN_simulator import *


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

    # 结果统计
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


# ---------- CMSIS-NN 推理所需参数导出 ---------- #

# 将 weight、bias 导出为 C数组 #

# export_tensor_as_c_array(conv1_simulator.weight_int8, "./parameter/LeNet/", "conv1_kernel", "conv1_kernel", "int8_t")
# export_tensor_as_c_array(conv1_simulator.bias_int32, "./parameter/LeNet/", "conv1_bias", "conv1_bias", "int32_t")
# export_tensor_as_c_array(conv2_simulator.weight_int8, "./parameter/LeNet/", "conv2_kernel", "conv2_kernel", "int8_t")
# export_tensor_as_c_array(conv2_simulator.bias_int32, "./parameter/LeNet/", "conv2_bias", "conv2_bias", "int32_t")

# export_tensor_as_c_array(fc1_simulator.weight_int8, "./parameter/LeNet/", "fc1_weight", "fc1_weight", "int8_t")
# export_tensor_as_c_array(fc1_simulator.bias_int32, "./parameter/LeNet/", "fc1_bias", "fc1_bias", "int32_t")
# export_tensor_as_c_array(fc2_simulator.weight_int8, "./parameter/LeNet/", "fc2_weight", "fc2_weight", "int8_t")
# export_tensor_as_c_array(fc2_simulator.bias_int32, "./parameter/LeNet/", "fc2_bias", "fc2_bias", "int32_t")
# export_tensor_as_c_array(fc3_simulator.weight_int8, "./parameter/LeNet/", "fc3_weight", "fc3_weight", "int8_t")
# export_tensor_as_c_array(fc3_simulator.bias_int32, "./parameter/LeNet/", "fc3_bias", "fc3_bias", "int32_t")

# print 反量化参数 #

conv1_simulator.print_multi_shift_zero_point()
conv2_simulator.print_multi_shift_zero_point()
fc1_simulator.print_multi_shift_zero_point()
fc2_simulator.print_multi_shift_zero_point()
fc3_simulator.print_multi_shift_zero_point()




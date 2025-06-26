import torch

from utils.mnist_util import get_DataLoader
from utils.params_util import load_state_dict_2_model


def generate_q_model(QuantizedModel, state_dict_path):

    # ------------ MNIST Loader & Quantized Model ------------ # 

    _, test_loader = get_DataLoader(64)

    model = load_state_dict_2_model(QuantizedModel, state_dict_path)
    model.eval()

    # ------------ Quantization Config ------------ # 

    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    model.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.default_observer.with_args(
            quant_min=0, quant_max=127),
        weight=torch.quantization.default_weight_observer.with_args(
            quant_min=-127, quant_max=127))
    
    model = torch.quantization.fuse_modules(model, [['hidden_layer', 'relu']])  
    model = torch.quantization.prepare(model, inplace=False)    

    # ------------ 校准 ------------ #

    with torch.no_grad():
        for data, _ in test_loader:
            model(data.to('cpu'))

    # ------------ 转化为量化模型 ------------ #

    q_model = torch.quantization.convert(model, inplace=False)

    return q_model


def generate_q_LeNet(QuantizedModel, state_dict_path):
    """
    用于将 LeNet 模型转为量化版本

    Args:
        QuantizedModel: 与 LeNet 匹配的量化模型
        state_dict_path: LeNet 模型参数保存地址

    Returns:
        q_model: 量化版本的 LeNet 模型
    """

    # ------------ MNIST Loader &  Quantized Model ------------ # 

    _, test_loader = get_DataLoader(64)

    model = load_state_dict_2_model(QuantizedModel, state_dict_path)
    model.eval()

    # ------------ Quantization Config ------------ # 

    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    model = torch.quantization.fuse_modules(model, [['conv1', 'relu1'], 
                                                    ['conv2', 'relu2'], 
                                                    ['fc1', 'relu3'], 
                                                    ['fc2', 'relu4']])
    model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer.with_args(
                quant_min=0, quant_max=127),
            weight=torch.quantization.default_weight_observer.with_args(
                quant_min=-128, quant_max=127))
    
    model = torch.quantization.prepare(model, inplace=False)

    # ------------ 校准 ------------ #

    with torch.no_grad():
        for data, _ in test_loader:
            model(data.to('cpu'))

    # ------------ 转化为量化模型 ------------ #
    
    q_model = torch.quantization.convert(model, inplace=False)

    return q_model
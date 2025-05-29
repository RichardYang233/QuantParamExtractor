import torch

from dataset import *
from utils.state_dict_utils import load_state_dict_2_model


def generate_q_model(QuantizedModel, state_dict_path):

    # ------------ MNIST Loader &  Quantized Model ------------ # 

    _, test_loader = get_DataLoader(64)

    model = load_state_dict_2_model(QuantizedModel, state_dict_path)
    model.eval()

    # ------------ Quantization Config ------------ # 

    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    model.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.default_observer.with_args(
            quant_min=0,          
            quant_max=127         
        ),
        weight=torch.quantization.default_weight_observer.with_args(
            quant_min=-127,        
            quant_max=127          
        )
    )
    model = torch.quantization.fuse_modules(model, [['hidden_layer', 'relu']])  
    model = torch.quantization.prepare(model, inplace=False)    

    # ------------ 校准 ------------ #

    with torch.no_grad():
        for data, _ in test_loader:
            model(data.to('cpu'))

    # ------------ 转化为量化模型 ------------ #

    q_model = torch.quantization.convert(model, inplace=False)

    return q_model
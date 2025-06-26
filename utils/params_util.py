import torch
import os
import csv

from train import *


# ------ 参数路径 ------ #

# 原始参数 (.pt)

FCNET_PARAMS_SAVE_PATH = './parameter/FCNet/FCNet_params.pt'
LENET_PARAMS_SAVE_PATH = './parameter/LeNet/LeNet_params.pt'

def save_state_dict(state_dict: dict, path: str) -> None:
    '''
    保存模型参数到指定路径
    Args:
        state_dict: 由 model.state_dict() 获取
        path: 保存路径   
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)

def load_state_dict_2_model(model, path: str):
    
    model = model.to(device=DEVICE)
    best_checkpoint_state_dict = torch.load(path, weights_only=True)
    model.load_state_dict(best_checkpoint_state_dict)

    return model

def _get_tensor_size(tensor):

    dim = tensor.dim()
    if dim == 1:
        return (tensor.size(0), None, None, None)
    if dim == 2:
        return (tensor.size(0), tensor.size(1), None, None)
    if dim == 4:
        return (tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3))

def export_tensor_as_c_array(tensor: torch.tensor, path, fileName, arrayName, type: str):
    '''
    type 仅支持: int8_t, int32_t
    '''
    if type not in ['int8_t', 'int32_t']:
        raise ValueError(f"type '{type}' 不支持，仅支持 'int8_t' 或 'int32_t'")

    dim = _get_tensor_size(tensor)

    if tensor.dim() == 4:
        tensor = tensor.permute(0, 2, 3, 1).contiguous().view(-1)   # 这里换维度很重要，以符合CMSIS-NN计算标准
    else:
        tensor = tensor.contiguous().view(-1)

    length = tensor.numel()

    with open(f"{path}{fileName}.h", "w") as f:
        # 头文件
        f.write(f"#ifndef {fileName.upper()}_H\n#define {fileName.upper()}_H\n\n")
        f.write(f"#include <stdint.h>\n\n")
        # 数组长度
        f.write(f"#define {arrayName.upper()}   {length}\n\n")
        # 尺寸说明
        f.write(f"// {dim[0]} * {dim[1]} * {dim[2]} * {dim[3]}\n")
        # 数组
        f.write(f"const {type} {arrayName}[{arrayName.upper()}] = {{\n")
        for i in range(length):
            f.write(f"{int(tensor[i])}, ")
            if (i + 1) % 50 == 0:
                f.write("\n")
        f.write("};\n\n#endif\n")




# 该函数暂时未使用
def pt2csv(src_path, drt_path=None):
    
    drt_path = src_path.replace('.pt', '.csv')

    best_checkpoint = dict()
    best_checkpoint['state_dict'] = torch.load(src_path, weights_only=True)

    with open(drt_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for name, params in best_checkpoint['state_dict'].items():
            writer.writerow([name, params.size()])

            params = params.cpu().numpy()

            if params.ndim == 1:
                writer.writerow(params.tolist())
            elif params.ndim == 2:
                writer.writerows(params.tolist())
            else:   # TODO: tensor dim=1 and dim=2
                continue
                # raise ValueError(f"Unexpected parameter dimensions: {params.ndim}")
            
    print(f'Successfully extract params from {src_path} to {drt_path} !!!') 
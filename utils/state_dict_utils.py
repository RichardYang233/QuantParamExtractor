import torch
import os
import csv

from train import *

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


def tensor_2_c_array(tensor: torch.tensor, fileName):
    
    tensor = tensor.view(-1)


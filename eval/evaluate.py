import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from train import *


# ------------ Accuracy ------------ # 

@torch.inference_mode() # 禁用梯度计算，节省内存并加速推理
                        # 等价于 torch.no_grad()
def evaluate(
  	model: nn.Module,
  	dataloader: DataLoader,
  	extra_preprocess = None
) -> float:
	model.eval()

	num_samples = 0
	num_correct = 0

	for inputs, targets in tqdm(dataloader, desc="eval", leave=False):

		if extra_preprocess is not None:
			for preprocess in extra_preprocess:
				inputs = preprocess(inputs)

		# Inference
		outputs = model(inputs.to('cpu'))
		predict = outputs.argmax(dim=1)

		# Update metrics
		num_samples += targets.size(0)
		num_correct += (predict == targets).sum()

	return (num_correct / num_samples * 100).item()

# ------------ Params ------------ # 

# 计算模型参数大小 （默认参数类型为 float32）
def get_model_size(model: nn.Module, data_width = 32):
    """
    calculate the model size in bits
    :param data_width: #bits per element
    """
    num_elements = 0;
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width
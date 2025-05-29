from torch import nn
from torch.utils.data import DataLoader
from torch.optim import *
from torch.optim.lr_scheduler import *
from tqdm.auto import tqdm


from .config import *




def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    callbacks = None
) -> None:
    
    model.train() # 模型设置为训练模式

    for inputs, targets in tqdm(dataloader, desc='train', leave=False):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad() # 梯度清零

        outputs = model.forward(inputs).cpu()
        loss = criterion(outputs, targets)

        loss.backward() # 计算梯度

        optimizer.step() # 更新参数

        if callbacks is not None:
            for callback in callbacks:
                callback()  






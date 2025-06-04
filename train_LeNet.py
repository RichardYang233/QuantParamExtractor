import copy
import torch
from torchvision.transforms import *

from utils import * 
from models import *
from train import *
from dataset import *
from eval import *


# ----- dataset ----- #

train_loader, test_loader = get_DataLoader(BATCH)

# ----- model ----- #

model = LeNet().to(device=DEVICE)

# ----- configs ----- #

LeNet_config = configs["LeNet"]
lr = LeNet_config["lr"]
momentum = LeNet_config['momentum']
epoch = LeNet_config['epoch']
optimizer = torch.optim.SGD(params=model.parameters(),  lr=lr, momentum=momentum)  # lr学习率，momentum冲量
criterion = nn.CrossEntropyLoss()  # 交叉熵损失

# ----- train ----- #

best_accuracy = 0
best_checkpoint = dict()

for epoch_round in range(epoch):
    train(model, train_loader, criterion, optimizer)
    accuracy = evaluate(model, test_loader)
    is_best = accuracy > best_accuracy
    if is_best:
        best_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
        best_accuracy = accuracy

    print(f'Epoch{epoch_round+1:>2d} Accuracy {accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%')


# save model
params_save_path = './parameter/LeNet_params.pt'
save_state_dict(best_checkpoint['state_dict'], params_save_path)

# show best_checkpoint
print(f"=> loading best checkpoint")
model.load_state_dict(best_checkpoint['state_dict'])

model_accuracy = evaluate(model, test_loader)
print(f"Model has accuracy={model_accuracy:.2f}%")














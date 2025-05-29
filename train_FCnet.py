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

model = FCNet().to(device=DEVICE)

# ----- configs ----- #

FCNet_config = configs["FCNet"]
lr = FCNet_config["lr"]
epoch = FCNet_config["epoch"]
optimizer = torch.optim.RMSprop(params=model.parameters(), lr=lr)   
criterion = nn.CrossEntropyLoss()   # 损失函数（含Softmax转化为概率分布）

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


# save best_checkpoint
save_state_dict(best_checkpoint['state_dict'], FCNET_PARAMS_SAVE_PATH)

# show best_checkpoint
print(f"=> loading best checkpoint")
load_state_dict_2_model(model, FCNET_PARAMS_SAVE_PATH)

model_accuracy = evaluate(model, test_loader)
print(f"Model has accuracy={model_accuracy:.2f}%")

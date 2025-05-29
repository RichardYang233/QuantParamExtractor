import torch
from torch import nn

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.hidden_layer = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

class QuantizedFCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.hidden_layer = torch.nn.Linear(784, 512)
        self.relu = torch.nn.ReLU()
        self.output_layer = torch.nn.Linear(512, 10)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.quant(x)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.dequant(x)
        return x


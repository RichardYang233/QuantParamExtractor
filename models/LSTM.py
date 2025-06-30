import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(in_features=128, out_features=10);

    def forward(self, x):
        x = x.squeeze(1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x
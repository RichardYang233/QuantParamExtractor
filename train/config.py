import torch


# ----- config ----- #

DEVICE = torch.device("cpu")

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

# ----- Config of train ----- #

BATCH = 64

configs = {
    "LeNet": {
        "lr": 0.01,
        "momentum": 0.5,
        "epoch": 3,
        "optimizer": "SGD",
        "criterion": "CrossEntropyLoss"
    },

    "FCNet": {
        "lr": 1e-4,
        "momentum": None,
        "epoch": 10,
        "optimizer": "RMS",
        "criterion": "CrossEntropyLoss"
    },

    "LSTM": {
        "lr": 0.005,
        "momentum": 0.9,
        "epoch": 5,
        "optimizer": "SGD",
        "criterion": "CrossEntropyLoss"
    }
}
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_DataLoader(batch: int) -> DataLoader:

    # 获取数据集
    train_dataset = datasets.MNIST(root='./dataset/mnist', train=True, download=True, transform=transforms.ToTensor())  
    test_dataset = datasets.MNIST(root='./dataset/mnist', train=False, download=True, transform=transforms.ToTensor())  # train=True训练集，=False测试集

    # 设置 DataLoader
    batch_size = batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
    


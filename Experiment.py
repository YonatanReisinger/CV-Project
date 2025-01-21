import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

class Experiment:
    def __init__(self, model: nn.Module, batch_size: int, train=None,test=None):
        self.model = model
        if train is None and test is None:
            train_dataset = datasets.CIFAR10(root='./data', train=True)
            test_dataset = datasets.CIFAR10(root='./data', train=False)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        else:
            self.train_loader = train
            self.test_loader = test

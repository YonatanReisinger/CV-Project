import pytest
import torch
from torchvision import datasets, transforms
from CNN import CNN

def test_CNN_sanity():
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    images, labels = next(iter(dataloader))
    model = CNN()
    output = model(images)
    # Assertions to check the output
    assert output.shape == (64, 1), f"Expected output shape (64, 1), but got {output.shape}"
    assert (0 <= output).all() and (output <= 1).all(), "Output values should be between 0 and 1 (after sigmoid)"
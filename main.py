import torch
from torchvision import datasets, transforms
from CNN import CNN

def main():
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    images, labels = next(iter(dataloader))
    model = CNN()
    output = model(images)
    print("hello")




if __name__ == '__main__':
    main()
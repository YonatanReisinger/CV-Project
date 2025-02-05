import torchvision
import torch
from Experiment import Experiment
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights
from typing import List

def get_resnet():
    resnet = torchvision.models.resnet34(pretrained=True)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10)
    torch.nn.init.xavier_uniform_(resnet.fc.weight)
    return resnet

def get_loaders():
    # We need to resize the images given resnet takes input of image size >= 224
    IMAGE_SIZE = 224
    # These values are mostly used by researchers as found to very useful in fast convergence
    mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
    # https://pytorch.org/vision/stable/transforms.html
    # We can try various transformation for good generalization of model
    training_transform, test_transform = get_training_transform(IMAGE_SIZE, mean, std)

    # Load the data and transform the dataset
    train_dataset = datasets.CIFAR10(root='./data',
                                  train=True,
                                  download=True,
                                  transform=training_transform)
    validation_dataset = datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=test_transform)

    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    # Create train and validation batch for training
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, num_workers=4, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=100, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

    return train_loader, validation_loader, test_loader

def get_training_transform(image_size: int, mean: List[float], std: List[float]):


    training_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)),  # Resize the image in a 32X32 shape
         transforms.RandomRotation(20),  # Randomly rotate some images by 20 degrees
         transforms.RandomHorizontalFlip(0.1),  # Randomly horizontal flip the images
         transforms.ColorJitter(brightness=0.1,  # Randomly adjust color jitter of the images
                                contrast=0.1,
                                saturation=0.1),
         transforms.RandomAdjustSharpness(sharpness_factor=2,
                                          p=0.1),  # Randomly adjust sharpness
         transforms.ToTensor(),  # Converting image to tensor
         transforms.Normalize(mean, std),  # Normalizing with standard mean and standard deviation
         transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)])

    test_transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    return training_transform, test_transform

def resnet_experiment_1():
    resnet_model = get_resnet()
    train_loader, validation_loader, test_loader = get_loaders()
    exp = Experiment(model=resnet_model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=100,
                     epochs=10,
                     lr=0.1,
                     momentum=0.2,
                     optimizer_name="SGD",
                     train=train_loader,
                     val=validation_loader,
                     test=test_loader)

    exp()
    exp.to_pickle()
    print(exp)

def resnet_experiment_2():
    resnet_model = get_resnet()
    training_transform, test_transform = get_training_transform(224, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    exp = Experiment(model=resnet_model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=100,
                     epochs=10,
                     lr=0.1,
                     momentum=0.2,
                     optimizer_name="SGD",
                     training_transform=training_transform,
                     test_transform=test_transform)

    exp()
    exp.to_pickle()
    print(exp)

def resnet_experiment_3():
    resnet_model = get_resnet()
    training_transform, test_transform = get_training_transform(224, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    training_transform = test_transform
    exp = Experiment(model=resnet_model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=256,
                     epochs=10,
                     lr=0.1,
                     momentum=0.2,
                     optimizer_name="SGD",
                     training_transform=training_transform,
                     test_transform=test_transform)

    exp()
    exp.to_pickle()
    print(exp)

def resnet_experiment_4():
    resnet_model = get_resnet()
    exp = Experiment(model=resnet_model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=256,
                     epochs=10,
                     lr=0.1,
                     momentum=0.2,
                     optimizer_name="SGD")

    exp()
    exp.to_pickle()
    print(exp)

def resnet_experiment_5():
    resnet_model = get_resnet()
    training_transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])
    exp = Experiment(model=resnet_model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=256,
                     epochs=10,
                     lr=1,
                     momentum=0.2,
                     optimizer_name="SGD",
                     training_transform=training_transform)

    exp()
    exp.to_pickle()

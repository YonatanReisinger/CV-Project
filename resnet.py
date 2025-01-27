import torchvision
import torch
from Experiment import Experiment

def get_resnet():
    resnet = torchvision.models.resnet34(pretrained=True)
    resnet.fc = torch.nn.Linear(512, 10)
    torch.nn.init.xavier_uniform_(resnet.fc.weight)
    return resnet


def resnet_experiment():
    resnet_model = get_resnet()
    exp = Experiment(model=resnet_model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=100,
                     epochs=25,
                     lr=0.1,
                     momentum=0.2,
                     optimizer_name="SGD")

    exp()

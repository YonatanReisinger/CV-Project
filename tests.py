import pytest
import torch
from torchvision import datasets, transforms
from CNN import CNN
from Experiment import Experiment

@pytest.fixture()
def train_dataloader():
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader

@pytest.fixture()
def model():
    model = CNN(layers=[3, 10, 20, 40], kernel_sizes=[3, 5, 5], strides=[1, 2, 2], output_size=1)
    return model

@pytest.fixture()
def experiment(model):
    exp = Experiment(model=model, criterion=torch.nn.CrossEntropyLoss(), batch_size=64, epochs=10, lr=0.01)
    return exp

@pytest.mark.parametrize("output_size", [1, 10])
def test_CNN_with_sigmoid(train_dataloader, model, output_size):
    images, labels = next(iter(train_dataloader))
    model.output_size = output_size
    model.output_activation = torch.sigmoid
    output = model(images)
    # Assertions to check the output
    assert output.shape == (64, output_size), f"Expected output shape (64, {output_size}), but got {output.shape}"
    assert (0 <= output).all() and (output <= 1).all(), "Output values should be between 0 and 1 (after sigmoid)"

@pytest.mark.parametrize("output_size", [1, 10])
def test_CNN_ten_output_with_softmax(train_dataloader, model, output_size):
    images, labels = next(iter(train_dataloader))
    model.output_activation = torch.softmax
    model.output_size = output_size
    output = model(images)
    # Assertions to check the output
    assert output.shape == (64, output_size), f"Expected output shape (64, {output_size}), but got {output.shape}"
    assert (0 <= output).all() and (output <= 1).all(), "Output values should be between 0 and 1 (after softmax)"

def test_experiment_sanity(experiment):
    experiment.model.output_activation = torch.softmax
    experiment.model.output_size = 10
    experiment()
    print(f"\n--------{experiment.score}--------")
    # Assertions to check the output
    assert 0 <= experiment.score <= 100, f"Expected accuracy to be between 0 and 100, but got {experiment.score}"
    assert len(experiment.models_states) == 10, f"Expected 10 models states, but got {len(experiment.models_states)}"
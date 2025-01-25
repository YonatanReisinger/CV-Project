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
    model = CNN(convolution_layers=[3, 32, 64, 128], kernel_sizes=[3, 5, 5], strides=[1, 2, 2], output_size=10, paddings=[2,2,2])
    return model

def test_invalid_hidden_activations_number(model):
    with pytest.raises(ValueError):
        model = CNN(convolution_layers=[3, 10, 20, 40],
                    kernel_sizes=[3, 5, 5],
                    strides=[1, 2, 2],
                    output_size=10,
                    hidden_activations=[torch.relu, torch.relu],
                    paddings=[2,2,2])

def test_invalid_kernel_sizes(model):
    with pytest.raises(ValueError):
        model = CNN(convolution_layers=[3, 10, 20, 40],
                    kernel_sizes=[3, 5],
                    strides=[1, 2, 2],
                    output_size=10,
                    hidden_activations=[torch.relu, torch.relu, torch.relu],
                    paddings=[2,2,2])

def test_invalid_strides(model):
    with pytest.raises(ValueError):
        model = CNN(convolution_layers=[3, 10, 20, 40],
                    kernel_sizes=[3, 5, 5],
                    strides=[1, 2],
                    output_size=10,
                    hidden_activations=[torch.relu, torch.relu, torch.relu],
                    paddings=[2,2,2])

def test_invalid_paddings(model):
    with pytest.raises(ValueError):
        model = CNN(convolution_layers=[3, 10, 20, 40],
                    kernel_sizes=[3, 5, 5],
                    strides=[1, 2, 2],
                    output_size=10,
                    hidden_activations=[torch.relu, torch.relu, torch.relu],
                    paddings=[2,2])

def test_invalid_experiment(model):
    with pytest.raises(ValueError):
        exp = Experiment(model=model,
                         criterion=torch.nn.CrossEntropyLoss(),
                         batch_size=64,
                         epochs=2,
                         lr=0.01,
                         optimizer_name="adam")


@pytest.mark.parametrize("output_size", [1, 10])
@pytest.mark.parametrize("output_activation", [torch.softmax, None])
def test_CNN(train_dataloader, output_size, output_activation):
    model = CNN(convolution_layers=[3, 32, 64, 128],
                kernel_sizes=[3, 5, 5],
                strides=[1, 2, 2],
                output_size=output_size,
                output_activation=output_activation,
                paddings=[2,2,2])
    images, labels = next(iter(train_dataloader))
    output = model(images)
    # Assertions to check the output
    assert len(model.hidden) == 3
    assert output.shape == (64, output_size), f"Expected output shape (64, {output_size}), but got {output.shape}"
    if output_activation:
        assert (0 <= output).all() and (output <= 1).all(), "Output values should be between 0 and 1 (after softmax or sigmoid)"

@pytest.mark.parametrize("output_activation", [None])
def test_experiment_sanity(output_activation):
    model = CNN(convolution_layers=[3, 32, 64, 128],
                kernel_sizes=[5, 5, 5],
                strides=[1, 2, 2],
                fully_connected_layers=[None, 1000, 1000, 1000, 1000, 1000],
                output_size=10,
                output_activation=output_activation,
                paddings=[2,2,2])
    exp = Experiment(model=model,
                     criterion=torch.nn.CrossEntropyLoss(),
                     batch_size=100,
                     epochs=2,
                     lr=0.1,
                     optimizer_name="SGD")

    exp()
    exp.to_pickle(f"test_{exp.score}_{output_activation.__name__ if output_activation else "None"}_{exp.optimizer.__class__.__name__}.pkl")
    print(exp)
    # Assertions to check the output
    assert len(exp.TRAIN_LOSS), f"Expected {exp.epochs} loss scores, but got {len(exp.TRAIN_LOSS)}"
    assert 0 <= exp.score <= 100, f"Expected accuracy to be between 0 and 100, but got {exp.score}"
    assert len(exp.models_states) == exp.epochs, f"Expected {exp.epochs} models states, but got {len(exp.models_states)}"
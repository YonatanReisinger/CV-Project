import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1)
        self.cnn2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=2)
        self.cnn3 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=2)
        self.fc1 = None  # Placeholder for the fully connected layer, initialized dynamically

    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.cnn3(x)
        x = torch.relu(x)
        # Dynamically calculate the flattened size
        if self.fc1 is None:
            flattened_size = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(flattened_size, 1)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return torch.sigmoid(x)
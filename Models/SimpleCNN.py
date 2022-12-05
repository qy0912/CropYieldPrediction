import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Sigmoid
class Simple_CNN(nn.Module):
    def __init__(self):
        super(Simple_CNN, self).__init__()
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(10, 16, kernel_size=3, stride=1, padding=0),
            BatchNorm2d(16),
            ReLU(),
            Dropout(0.2),

            Conv2d(16, 8, kernel_size=3, stride=1, padding=0),
            BatchNorm2d(8),
            ReLU(),
            Dropout(0.2),
        )

        self.fc_layer = Sequential(
            Linear(2312,500),
            ReLU(),
            Dropout(0.2),
            Linear(500,1),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
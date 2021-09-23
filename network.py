import torch
import torch.nn as nn
import torch.optim as optimizer
import argparse
import torch.nn.functional as F
from utils import showImage, saveImage, showTensor, saveTensor

import numpy as np

class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(4, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v

class Qnetwork(nn.Module):
    def __init__(self, actionSpaceSize , alpha):
        super(Qnetwork, self).__init__()
        self.alpha = alpha
        self.conv1 = nn.Conv2d(1, 16, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(16, 32,4, stride=2)
        # TODO flattening hier ? 
        self.fc1 = nn.Linear(32*10*10, 256)
        self.fc2 = nn.Linear(256, actionSpaceSize)

        self.optimizer = optimizer.RMSprop(self.parameters(), lr=self.alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input):
        # showImage(input)
        self.printSize(input[:], "before list")
        input = list(input) # convert to list to accomidate pyTorch's format
        self.printSize(input[:], "after list")
        # print(input[:])
        prop = torch.Tensor(input).to(self.device)
        self.printSize(prop, "prop")
        # saveTensor(prop)

        prop = prop.view(-1, 1, 96, 96)
        print(len(prop[0][0]))

        # CONVOLUTION LAYERS

        prop = F.relu(self.conv1(prop))
        print(len(prop[0][0]))

        prop = F.relu(self.conv2(prop))
        print(len(prop[0][0]))

        print("output length of conv2: \n" + str(len(prop[0])) + "x" + str(len(prop[0][0])) + "x" + str(len(prop[0][0][0])) )

        # FLATTENING
        flat = prop.view(-1, 32 * 10 * 10)

        # FULLY CONNECTED
        prop = F.relu(self.fc1(flat))
        out = self.fc2(prop)
        # print(out)

        return out


    def printSize(self, observation, message = "<message>"):
        print(message + " - input size of is: " + str(len(observation)) + " x " + str(len(observation[0])))
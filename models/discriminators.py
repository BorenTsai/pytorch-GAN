import torch
import torch.nn as nn
import numpy as np

from models.utils import *

class VanillaD(nn.Module):
    """
    The GAN that started it all
    https://arxiv.org/pdf/1406.2661.pdf
    """
    def __init__(self,
                 img_size=[28, 28, 1],
                 out_dim=1,
                 hidden_layers=[1024, 512, 256],
                 dropout=0.5,
                 hidden_nonlinearity=nn.LeakyReLU,
                 out_nonlinearity=nn.Sigmoid,
                 ):
        super(VanillaD, self).__init__()
        self.img_size = img_size
        self.input_dim = np.prod(self.img_size)
        self.out_dim = out_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.hidden_nonlinearity = hidden_nonlinearity
        self.out_nonlinearity = out_nonlinearity

        layers = [self.input_dim] + self.hidden_layers + [self.out_dim]

        
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model = MLP(layers,
                         hidden_nonlinearity=self.hidden_nonlinearity,
                         output_nonlinearity=self.out_nonlinearity,
                         dropout=self.dropout)

        if torch.cuda.is_available():
            self.model = self.model.cuda()


    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        out = self.model(x)
        return out


class CycleD(nn.Module):
    def __init__(self):
        super(CycleD, self).__init__()

        print('\nCreating CycleGAN Discriminator')
        self.create_network()

    def create_network(self):
        # 70x70 PatchGAN
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), #4 FOV
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), #10 FOV
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), #22 FOV
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1), #46 FOV
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1) #70 FOV
        )

    def forward(self, x):
        out = self.model(x)
        return out


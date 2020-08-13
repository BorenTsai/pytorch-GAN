import torch
import torch.nn as nn
import numpy as np

from models.utils import *


class VanillaG(nn.Module):
    """
    The GAN that started it all
    https://arxiv.org/pdf/1406.2661.pdf
    """

    def __init__(self,
                 z_dim,
                 img_size=[28, 28, 1],
                 hidden_layers=[256, 512, 1024],
                 hidden_nonlinearity=nn.LeakyReLU,
                 out_nonlinearity=nn.Tanh,
                 ):
        super(VanillaG, self).__init__()
        self.z_dim = z_dim
        self.img_size = img_size
        self.out_dim = np.prod(self.img_size)
        self.hidden_layers = hidden_layers
        self.hidden_nonlinearity = hidden_nonlinearity
        self.out_nonlinearity = out_nonlinearity

        layers = [self.z_dim] + self.hidden_layers + [self.out_dim]

        self.model = MLP(layers,
                         hidden_nonlinearity=self.hidden_nonlinearity,
                         output_nonlinearity=self.out_nonlinearity)

        init_weights(self.model)

    def forward(self, x):
        out = self.model(x)
        return out

    def generate_samples(self, num_samples):
        print('\nGenerating {} samples'.format(num_samples))
        noise = torch.randn(num_samples, self.z_dim)
        samples = self.model(noise)
        return samples.view(samples.size(0), self.img_size[2], self.img_size[0], self.img_size[1])


class CycleG(nn.Module):
    def __init__(self,
                 img_size=[128, 128, 3],
                 padding='reflection'):
        super(CycleG, self).__init__()
        self.img_size = img_size
        self.padding = padding
        self.num_blocks = 6 if self.img_size[0] == 128 else 9

        print('\nCreating CycleGAN Generator')
        self.create_network()

    def create_network(self):

        down_sampler = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        residual_blocks = []
        for _ in range(self.num_blocks):
            residual_blocks.append(ResidualBlock(256, padding=self.padding, ))
        residual_blocks = nn.Sequential(*residual_blocks)

        upsampler = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(3),
            nn.Tanh()
        )

        modules = [down_sampler] + [residual_blocks] + [upsampler]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x)
        return out

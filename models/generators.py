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

        self.model = create_mlp(layers,
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
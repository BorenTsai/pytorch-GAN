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

        self.model = create_mlp(layers,
                                hidden_nonlinearity=self.hidden_nonlinearity,
                                output_nonlinearity=self.out_nonlinearity,
                                dropout=self.dropout)
        #init_weights(self.model)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        out = self.model(x)
        return out

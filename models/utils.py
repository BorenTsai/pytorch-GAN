import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,
                 layers,
                 hidden_nonlinearity=nn.Tanh,
                 output_nonlinearity=nn.Identity,
                 dropout=0):
        super(MLP, self).__init__()
        self.layers = layers
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.dropout = dropout

        self.create_network()

    def create_network(self):
        modules = []
        for i in range(len(self.layers) - 2):
            if self.dropout > 0:
                modules.append(
                    nn.Sequential(nn.Linear(self.layers[i],
                                            self.layers[i + 1]),
                                            self.hidden_nonlinearity(),
                                            nn.Dropout(self.dropout)))
            else:
                modules.append(nn.Sequential(nn.Linear(self.layers[i], self.layers[i + 1]), self.hidden_nonlinearity()))
        modules.append(nn.Sequential(nn.Linear(self.layers[-2], self.layers[-1]), self.output_nonlinearity()))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x)
        return out

class CNN(nn.Module):
    def __init__(self,
                 channels,
                 kernels,
                 strides=None,
                 paddings=None,
                 poolings=None,
                 layer_norms=None,
                 nonlinearites=None,
                 name='CNN'):
        super(CNN, self).__init__()
        self.channels = channels
        self.kernels = kernels
        self.strides = strides if strides is not None else [1]*len(self.kernels)
        self.paddings = paddings if strides is not None else [0]*len(self.kernels)
        self.poolings = poolings
        self.layer_norms = layer_norms
        self.nonlinearites = nonlinearites if nonlinearites is not None else [nn.ReLU]*len(self.kernels[:-1]) + [nn.Identity]

        print('\n Creating {}'.format(name))
        self.create_network()

    def create_network(self):
        modules = []
        for i in range(len(self.kernels)):

            in_channels = self.channels[i]
            out_channels = self.channels[i+1]
            module = []

            module.append(nn.Conv2d(in_channels, out_channels, kernel_size=self.kernels[i], stride=self.strides[i], padding=self.paddings[i]))
            if self.poolings is not None:
                module.append(self.poolings[i]())
            if self.layer_norms is not None:
                module.append(self.layer_norms[i](out_channels))
            module.append(self.nonlinearites[i])

            module = nn.Sequential(*module)
            modules.append(module)

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self,
                 channels,
                 padding='reflection'):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.padding = 'reflection'

        self.create_network()

    def create_network(self):
        modules = []
        padding = 0
        if self.padding == 'reflection':
            modules.append(nn.ReflectionPad2d(1))
        # Assume 0 padding
        else:
            padding = 1

        modules.extend([
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=padding),
            nn.InstanceNorm2d(self.channels),
            nn.ReLU(inplace=True)
        ])

        if self.padding:
            modules.append(nn.ReflectionPad2d(1))

        modules.extend([
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=padding),
            nn.InstanceNorm2d(self.channels)
        ])

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x)
        return x + out


def create_mlp(layers, hidden_nonlinearity=nn.Tanh, output_nonlinearity=nn.Identity, dropout=0):
    modules = []
    for i in range(len(layers) - 2):
        if dropout > 0:
            modules.append(nn.Sequential(nn.Linear(layers[i], layers[i+1]), hidden_nonlinearity(), nn.Dropout(dropout)))
        else:
            modules.append(nn.Sequential(nn.Linear(layers[i], layers[i + 1]), hidden_nonlinearity()))
    modules.append(nn.Sequential(nn.Linear(layers[-2], layers[-1]), output_nonlinearity()))
    mlp = nn.Sequential(*modules)

    return mlp

def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
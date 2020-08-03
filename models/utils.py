import torch
import torch.nn as nn


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

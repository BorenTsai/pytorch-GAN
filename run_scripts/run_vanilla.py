import torch.nn as nn

from models.discriminators import VanillaD
from models.generators import VanillaG
from loaders.torchvision_datasets import *
from trainers.vanilla_trainer import Trainer

def run_experiment(**kwargs):

    generator = VanillaG(
        z_dim=kwargs['z_dim'],
        hidden_layers=kwargs['g_hidden_layers'],
        hidden_nonlinearity=kwargs['g_hidden_nonlinearity'],
        out_nonlinearity=kwargs['g_out_nonlinearity'],
        img_size=kwargs['img_size']
    )

    discriminator = VanillaD(
        img_size=kwargs['img_size'],
        out_dim=kwargs['d_out_dim'],
        hidden_layers=kwargs['d_hidden_layers'],
        dropout=kwargs['dropout'],
        hidden_nonlinearity=kwargs['d_hidden_nonlinearity'],
        out_nonlinearity=kwargs['d_out_nonlinearity'],
    )


    if kwargs['data_type'] == 'mnist':
        dataloader = get_MNIST_loader(kwargs['batch_size'])
    elif kwargs['data_type'] == 'kmnist':
        dataloader = get_KMNIST_loader(kwargs['batch_size'])
    else:
        raise NotImplementedError



    T = Trainer(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        k=kwargs['k'],
        lr=kwargs['lr'],
        num_epochs=kwargs['num_epochs'],
        num_samples=kwargs['num_samples'],
        log_dir=kwargs['log_dir'],
        gpu_id=kwargs['gpu_id']
    )

    T.train()

if __name__ == '__main__':
    img_size = [28, 28, 1]

    config = {
        'gpu_id': 0,
        'data_type': 'kmnist',
        'img_size': img_size,

        # Generator Params
        'z_dim': 100,
        'g_hidden_layers': [256, 512, 1024],
        'g_hidden_nonlinearity': nn.LeakyReLU,
        'g_out_nonlinearity': nn.Tanh,

        # Discriminator Params
        'd_out_dim': 1,
        'd_hidden_layers': [1024, 512, 256],
        'dropout': 0.4,
        'd_hidden_nonlinearity': nn.LeakyReLU,
        'd_out_nonlinearity': nn.Sigmoid,

        # DataLoader Params
        'batch_size': 128,

        # Trainer Params
        'k': 1,
        'lr': 5e-4,
        'num_epochs': 1000,
        'num_samples': 10,
        'log_dir': 'experiments/VanillaGAN_KMNIST/1'
    }

    run_experiment(**config)

import torch.nn

from models.generators import CycleG
from models.discriminators import CycleD
from loaders.cycleGAN_dataloader import *
from trainers.cycle_trainer import Trainer

def run_experiment(**kwargs):

    G = CycleG(
        img_size=kwargs['img_size'],
        padding=kwargs['padding']
    )

    F = CycleG(
        img_size=kwargs['img_size'],
        padding=kwargs['padding']
    )

    Dx = CycleD()
    Dy = CycleD()

    train_dataset = CycleGANDataset(
        from_dir=kwargs['train_from_dir'],
        to_dir=kwargs['train_to_dir'],
        im_size=kwargs['img_size']
    )

    test_dataset = CycleGANDataset(
        from_dir=kwargs['test_from_dir'],
        to_dir=kwargs['test_to_dir'],
        im_size=kwargs['img_size']
    )

    train_dataloader = DataLoader(
            dataset=train_dataset,
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1
    )

    test_dataloader = DataLoader(
            dataset=test_dataset,
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=1
    )

    T = Trainer(
        G=G,
        F=F,
        Dx=Dx,
        Dy=Dy,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        lr=kwargs['lr'],
        num_epochs=kwargs['num_epochs'],
        gpu_id=kwargs['gpu_id'],
        log_dir=kwargs['log_dir']
    )

    T.train()

if __name__ == '__main__':
    config = {
        'gpu_id': 0,
        'log_dir': './experiments/CycleGAN_test',
        'num_epochs': 100,

        'img_size': [256, 256],
        'padding': 'reflection',
        
        'train_from_dir': './data/cycleGAN/summer2winter_yosemite/trainA',
        'train_to_dir': './data/cycleGAN/summer2winter_yosemite/trainB',
        'test_from_dir': './data/cycleGAN/summer2winter_yosemite/testA',
        'test_to_dir': './data/cycleGAN/summer2winter_yosemite/testB',

        'lr': 2e-4,
        'batch_size': 1,

    }

    run_experiment(**config)

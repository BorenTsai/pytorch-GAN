import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def get_MNIST_loader(batch_size):
    tlist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data = torchvision.datasets.MNIST(root='./data',
                                      train=True,
                                      download=True,
                                      transform=tlist)

    loader = DataLoader(dataset=data,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=1)

    return loader


def get_KMNIST_loader(batch_size):
    tlist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data = torchvision.datasets.KMNIST(root='./data',
                                      train=True,
                                      download=True,
                                      transform=tlist)

    loader = DataLoader(dataset=data,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=1)

    return loader

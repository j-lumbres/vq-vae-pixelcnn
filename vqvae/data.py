import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_cifar10_loaders(data_dir, batch_size, num_workers=2):
    # Normalize every channel to [-1, 1] so it matches the decoder's tanh output
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Windows starts each worker by re-importing the main script, and the training scripts
    # have no __main__ guard, so every worker would start training too. Load in-process there.
    if sys.platform == 'win32':
        num_workers = 0

    train_data = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

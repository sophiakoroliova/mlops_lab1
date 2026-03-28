import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict, Any


def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepares and returns training, validation, and test DataLoaders."""

    # Define image transformations: convert to Tensor and normalize pixels to [-1, 1] range
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the CIFAR-10 dataset
    full_train = datasets.CIFAR10(root=config['data']['data_dir'], train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=config['data']['data_dir'], train=False, download=True, transform=transform)

    # Split the training data into Training and Validation sets based on config['data']['val_size']
    train_size = int((1 - config['data']['val_size']) * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    # Create DataLoaders for efficient batch processing and shuffling
    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config['training']['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader
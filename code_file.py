import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets

training_data = datasets.CIFAR10(
    root = "Data",
    train = True,
    download = True,
    transform = ToTensor()
)

testing_data = datasets.CIFAR10(
    root = "Data",
    train = False,
    download = True,
    transform = ToTensor()
)
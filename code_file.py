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

batch_size = 64

train_loader = DataLoader(dataset = training_data, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(dataset = testing_data, batch_size = batch_size, shuffle=False)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
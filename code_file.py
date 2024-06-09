import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

num_epochs = 11
batch_size = 10
learning_rate = 0.005

transform_train = transforms.Compose([transforms.ToTensor(),
                                      transforms.RandomHorizontalFlip(), transforms.RandomCrop(32),
                                      transforms.Grayscale(),
                                      transforms.Normalize(0.5, 0.5)])

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Grayscale(),
                                     transforms.Normalize(0.5, 0.5)])

train_dataset = datasets.CIFAR10(
    root = 'Data',
    train = True,
    download = True,
    transform=transform_train
)

test_dataset = datasets.CIFAR10(
    root = 'Data',
    train = False,
    download = True,
    transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.r = nn.ReLU()
        self.pool = nn.AvgPool2d(2,2)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.r(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.r(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.r(x)
        x = self.fc2(x)
        x = self.r(x)
        x = self.fc3(x)
        return(x)

model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print(f"Epoch : [ {epoch+1}/{num_epochs} ], Step [ {i+1}/{num_steps} ], Loss: {loss.item}")

print('Finished Training!')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for X,y in test_loader:
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        score, prediction = torch.max(pred, 1)
        n_samples += y.size(0)
        n_correct += (prediction == y).sum().item()

        
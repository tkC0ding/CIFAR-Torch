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

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.r1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 4)
        self.r2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 32, 3)
        self.r3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 64)
        self.r4 = nn.ReLU()
        self.fc2 = nn.Linear(64, 16)
        self.r5 = nn.ReLU()
        self.fc3 = nn.Linear(16, 10)

        nn.ModuleList([self.conv1, self.r1, self.pool1, self.conv2, self.r2, self.pool2, self.conv3, self.r3, self.pool3,
                       self.flatten, self.fc1, self.r4, self.fc2, self.r5, self.fc3])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.r1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.r2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.r3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.r4(x)
        x = self.fc2(x)
        x = self.r5(x)
        x = self.fc3(x)
        return(x)

model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
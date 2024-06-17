import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

batch_size = 64
learning_rate = 0.001
num_epochs = 5

# Choose device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Downloading datasets
train_dataset = datasets.CIFAR10(
    root='Data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

test_dataset = datasets.CIFAR10(
    root='Data',
    train = False,
    download = True,
    transform = transforms.ToTensor()
)

# Creating data loaders
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

class CNN:
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.5)
        self.flatten = nn.Flatten()
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.Dense1 = nn.Linear(1024, 4096)
        self.Dense2 = nn.Linear(4096, 4096)
        self.Dense3 = nn.Linear(4096, 10)

        nn.ModuleList([self.relu, self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.Dense1, self.Dense2,
                       self.Dense3, self.dropout, self.flatten, self.pool])


# train function
def train(model, loss_fn, optimizer, dataloader):
    model.train()
    num_steps = len(dataloader)
    for epoch in range(num_epochs):
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 782 == 0:
                print(f"Epoch [ {epoch+1}/{num_epochs} ] Step [ {i+1}/{num_steps} ] Loss: {loss.item()}")
    print("Training Finished!")


# validation function
def validate(model, dataloader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            scores, predictions = torch.max(pred, 1)
            n_correct += (predictions == y).sum().item()
            n_samples += y.size(0)
        acc = (n_correct/n_samples)*100
        print("Accuracy :", acc)

train(model, loss_fn, optimizer, train_loader)
validate(model, test_loader)
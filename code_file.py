import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

batch_size = 100
learning_rate = 0.001
num_epochs = 100

#define classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Choose device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

#Data Transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomInvert(),
    transforms.RandomRotation(10),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Downloading datasets
train_dataset = datasets.CIFAR10(
    root='Data',
    train = True,
    download = True,
    transform = train_transform
)

test_dataset = datasets.CIFAR10(
    root='Data',
    train = False,
    download = True,
    transform = test_transform
)

# Creating data loaders
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv6 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv7 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        self.Dense1 = nn.Linear(512, 128)
        self.Dense2 = nn.Linear(128, 64)
        self.Dense3 = nn.Linear(64, 16)
        self.Dense4 = nn.Linear(16, 10)

        nn.ModuleList([self.relu, self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.Dense1, self.Dense2,
                       self.Dense3, self.dropout, self.flatten, self.pool, self.conv7, self.conv8, self.Dense4])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv7(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.Dense1(x)
        x = self.relu(x)
        x = self.Dense2(x)
        x = self.relu(x)
        x = self.Dense3(x)
        x = self.relu(x)
        out = self.Dense4(x)
        return(out)

model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train function
def train(model, loss_fn, optimizer, dataloader):
    num_samples = 0
    num_correct = 0
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

            _, pred = torch.max(y_pred, 1)
            num_correct += (pred == y).sum().item()
            num_samples += y.size(0)
            acc = (num_correct/num_samples) * 100

            if (i + 1) % 250 == 0:
                print(f"Epoch [ {epoch+1}/{num_epochs} ] Step [ {i+1}/{num_steps} ] Loss: {loss.item()} Accuracy:{acc}")
    print("Training Finished!")


# validation function
def validate(model, dataloader):
    with torch.no_grad():
        model.eval()
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
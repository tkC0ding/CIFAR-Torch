import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
import torchvision.transforms as transforms

batch_size = 64
learning_rate = 0.001
num_epochs = 5

# Choose device
device = torch.device(
    "cpu"
)

#Data preprocessing
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize(256)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.Resize(256)
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

# Loading alexnet
model = models.alexnet(weights=True).to(device)

# Freezing model parameters
for params in model.parameters():
    params.requires_grad = False

# Change the final layer
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 10)

# Compile
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Using {device} device.")

print(model)

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
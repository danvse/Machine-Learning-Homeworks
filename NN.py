from tabulate import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time

# Ensure only GPU is used
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This code must be run on a GPU-enabled machine.")
device = torch.device("cuda")

print(f"Using device: {torch.cuda.get_device_name(0)}")  # Optional debug info

# Data Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# CNN + LSTM Model
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # output: (32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2),                            # (32, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2)                             # (64, 8, 8)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)                                 # (B, 64, 8, 8)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, 8*8, 64)  # (B, 64, 64)
        lstm_out, _ = self.lstm(x)                      # (B, 64, H)
        out = lstm_out[:, -1, :]                        # Take last output
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# Training and evaluation


def train_and_evaluate(hidden_size, optimizer_type, num_epochs):
    start_time = time.time()

    model = CNN_LSTM(hidden_size).to(device)
    criterion = nn.CrossEntropyLoss().to(device)  # <- Just to be consistent

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Accuracy calculation
    def evaluate(loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    train_acc = evaluate(train_loader)
    test_acc = evaluate(test_loader)

    duration = time.time() - start_time
    print(f"Done in {duration:.2f} seconds\n")

    return train_acc, test_acc

results = []
for hidden_size in [100, 200]:
    for optimizer in ['adam', 'sgd']:
        for epochs in [50, 100]:
            print(f"Training with hidden_size={hidden_size}, optimizer={optimizer}, epochs={epochs}")
            train_acc, test_acc = train_and_evaluate(hidden_size, optimizer, epochs)
            results.append((hidden_size, optimizer, epochs, train_acc, test_acc))

headers = ["Hidden Size", "Optimizer", "Epochs", "Train Acc (%)", "Test Acc (%)"]
print(tabulate(results, headers=headers, floatfmt=".2f"))

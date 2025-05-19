import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameters 
num_epochs = 5
batch_size = 4
learning_rate = 0.001

#loading the dataset, in here was are loading 10 dataset that was avalible from pytorch, which is why for the train/test, we are using torch to call the dataset modules to load data to automically load and optimize data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
#convolutional network implementation using cross entrophy
class ConvNet(nn.Module):
    #implement 
    def __init__(self):
        super(ConvNet, self).__init__()
        #input is 3, and the inputs are based on a matrix representation of the input pool array given
        self.conv1 = nn.Conv2d(3, 6, 5)
        #2x2 matrix pool
        self.pool = nn.MaxPool2d(2, 2)
        #second convol layer
        self.conv2 = nn.Conv2d(6, 16, 5)
        #connects the convolutional layer, input and output, can be different output
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #2nd fully connected layer
        self.fc2 = nn.Linear(120, 84)
        #third convolution layer, this value here in parameter HAS TO BE FIXED
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        #activation function and convolutional layer
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        #2nd activation and colutional layer
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        #flattening procress, define the correct size for flattening
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        #fully connected layers to a activation functions
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x

#creates our model
model = ConvNet().to(device)
#creates our loss function optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#trainingg loop, loop to get number of patches
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass, creates the loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize, empty the gradient first
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)
#evaluates the model, does not need gradient here
with torch.no_grad():
    #calculating accuracy
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
#calculated the accuracy for the entire network
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')


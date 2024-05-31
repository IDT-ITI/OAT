"""
This script regularly trains and evaluates an EfficientNet on the BAR dataset.

The script includes the following functionalities:
1. Parses command-line arguments for the configurable parameters (number of epochs, learning rate, batch size, and number of classes).
2. Loads the BAR dataset using ImageFolder Class. Note: set the appropriate paths.
3. Defines the efficientnet-b1 model.
4. Trains the model and evaluates the performance in terms of test accuracy after each epoch.

Note: Please ensure you have installed the efficientnet_pytorch package before running the script:
    pip3 install efficientnet_pytorch

Usage:
    python3 baseline_efficientnet.py --nof_epochs 100 --l_r 0.001 --batch_size 32 --nof_classes 6
"""

#--------------------------------------------------
# Imports
#-------------------------------------------------- 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from efficientnet_pytorch import EfficientNet
import argparse

# Set the device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#--------------------------------------------------
# Hyperparameters
#-------------------------------------------------- 
# Argument parser for number of epochs, learning rate, batch size, and number of classes
parser = argparse.ArgumentParser(description='Train Efficient using OAT.')
parser.add_argument('--nof_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--l_r', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--nof_classes', type=int, default=6, help='number of classes')

args = parser.parse_args()

# Hyperparameters
n_of_epochs = args.nof_epochs
learning_rate = args.l_r
batchsize = args.batch_size
nClasses = args.nof_classes


#--------------------------------------------------
# Dataset
#-------------------------------------------------- 
transform = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


trainset = torchvision.datasets.ImageFolder(root='/path/to/BAR/train/', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.ImageFolder(root='/path/to/BAR/test/', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                         shuffle=True, num_workers=0)


#--------------------------------------------------
# EfficientNet-b1 model
#-------------------------------------------------- 
net = EfficientNet.from_pretrained('efficientnet-b1')
net._fc = nn.Linear(net._fc.in_features, nClasses)  
net._fc = net._fc.to(device)
net = net.to(device)


#--------------------------------------------------
# Loss function and optimizer
#-------------------------------------------------- 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr= learning_rate, momentum=0.9)


#---------------------------------------------
#  Train the network
#---------------------------------------------
def train(epoch):
    net.train()

    for batch_idx, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


#---------------------------------------------
# Test the network
#---------------------------------------------
def test():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Epoch {}, Test accuracy: {:.3f}'.format(epoch, 100. * correct / len(testloader.dataset)))


for epoch in range(1, n_of_epochs + 1):
    train(epoch)
    test()

#---------------------------------------------
# Save the trained model
#---------------------------------------------
print('Saving the trained model.')
torch.save(net.state_dict(), 'baseline_efficientnet.pth')

"""
This script trains and evaluates an EfficientNet on the BAR dataset using the OAT method.

The script includes the following functionalities:
1. Parses command-line arguments for the configurable parameters (number of epochs, learning rate, batch size, number of classes, ptraining step, path to pretrained weights).
2. Loads the BAR dataset using ImageFolder Class. Note: set the appropriate paths.
3. Defines the efficientnet-b1 model.
4. Trains the model according to the OAT method (with or without pretrained weights) and evaluates the performance in terms of test accuracy after each epoch.

Note: Please ensure you have installed the efficientnet_pytorch package before running the script:
    pip3 install efficientnet_pytorch

Usage:
    python3 oat_efficientnet.py --nof_epochs 90 --l_r 0.001 --batch_size 32 --nof_classes 6 --pretrained True --weights_path /path/to/efficient10.pth
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
# Argument parser for number of epochs, learning rate, batch size, number of classes, pretraining step, and path to pretrained weights
parser = argparse.ArgumentParser(description='Train Efficient using OAT.')
parser.add_argument('--nof_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--l_r', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--nof_classes', type=int, default=6, help='number of classes')
parser.add_argument('--pretrained', type=bool, default=False, help='use pretrained weights')
parser.add_argument('--weights_path', type=str, default='', help='path to pretrained weights')

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


trainset = torchvision.datasets.ImageFolder(root='/path/to/train/', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.ImageFolder(root='/path/to/test/', transform=transform)
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
# Loading pretrained weights
#-------------------------------------------------- 
if args.pretrained:
    try:
        net.load_state_dict(torch.load(args.weights_path))
        print('Loaded pretrained weights. OAT training with a pretraning step.')
    except FileNotFoundError:
        print('Pretrained weights not found. OAT training from scratch.')
else:
    print('OAT training from scratch.')


#--------------------------------------------------
# Loss function and optimizer
#-------------------------------------------------- 
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr= learning_rate, momentum=0.9)


#--------------------------------------------------
# Compute anchors on the fly
#--------------------------------------------------
def batch_anchor(input):
    anchor = input.mean(0)
    return nn.functional.softmax(anchor,dim=-1)


#--------------------------------------------------
# Compute anchored targets
#--------------------------------------------------
def compute_targets(labels,anchor_train_batch):
    one_hot = torch.nn.functional.one_hot(labels, num_classes=nClasses).cpu()
    anchor_train_batch_cpu = anchor_train_batch.cpu()
    target = torch.zeros(labels.shape[0], nClasses)
    target = one_hot/anchor_train_batch_cpu - 1
    return target


#--------------------------------------------------
# Reverse output
#--------------------------------------------------
def reverse_output(output,anchor_test_batch):
    reverse_output = torch.zeros(output.shape[0], output.shape[1])
    reverse_output = anchor_test_batch*(output + 1)
    return reverse_output


#---------------------------------------------
#  Train the network
#---------------------------------------------
def train(epoch):
    net.train()

    for batch_idx, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        anchor_train_batch = batch_anchor(nn.functional.softmax(outputs,dim=1)).to(device)
        anchored_targets = compute_targets(labels,anchor_train_batch).to(device)
        loss = criterion(outputs, anchored_targets)
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
            anchor_test_batch = batch_anchor(nn.functional.softmax(outputs,dim=1)).to(device)
            outputs_reverse = reverse_output(outputs,anchor_test_batch).to(device)
            _, predicted = torch.max(outputs_reverse.data, 1)
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
torch.save(net.state_dict(), 'oat_efficientnet.pth')

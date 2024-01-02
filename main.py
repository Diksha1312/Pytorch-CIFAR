import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torchvision.models import resnet18

import numpy as np

import train, test

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Preparing data
# if not os.path.exists("data"):
#     os.makedirs("data")
     
# DATA_DIR = "data/"

mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])
batch_size = 100

print("===> Preparing data")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

trainset = torchvision.datasets.CIFAR10(root="./data", download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', download=False, transform=transform_test)

# Dataloading

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')

# for inputs,labels in trainloader:
#     print(inputs.shape)
#     print(labels.shape)
#     break

# print(len(trainloader))
# print(len(testloader))

# Model

print("===> Building model")

model = resnet18()
model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True


# Resume if you already have model

if os.path.exists('model/resnet.pth'):
    print("===> Resuming from checkpoint")
    checkpoint = torch.load('model/resnet.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(best_acc)
    print(start_epoch)
else:
    start_epoch = 0
    best_acc = 0.0

#Param initialisation

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay = 5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


num_epochs = 10

for epoch in range(start_epoch, num_epochs):
    train.train(epoch, num_epochs, model, trainloader, device, optimizer, criterion)
    acc = test.test(epoch, num_epochs, model, testloader, device, criterion)
    scheduler.step()

    if acc > best_acc:
        print("===> Saving model")
        state = {
            'model' : model.state_dict(),
            'acc' : acc,
            'epoch': epoch,
        }
        if not os.path.isdir('model'):
            os.makedirs('model')
        torch.save(state, './model/resnet.pth')
        best_acc = acc









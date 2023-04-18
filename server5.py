import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.nn import functional as F
import sys
import os
import math
import random

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
print(device)


# Hyperparameter configurations
exp_num = 5     # To save the result, change every time. -1 to not save.
total_epoch = 1000
learning_rate = 0.001
dropout_prob = 0.5
top_k = 1
if exp_num!=-1: os.mkdir(f'./Result{exp_num}')


def dataset_MNIST():
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
  train_dataset = MNIST(root='../data', train=True, download=True, transform=transform)
  test_dataset = MNIST(root='../data', train=False, download=True, transform=transform)
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)
  return train_loader, test_loader

def dataset_FMNIST():
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.2860,), (0.3530,))])
  train_dataset = FashionMNIST(root='../data', train=True, download=True, transform=transform)
  test_dataset = FashionMNIST(root='../data', train=False, download=True, transform=transform)
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)
  return train_loader, test_loader

def dataset_CIFAR10():
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
  train_dataset = CIFAR10(root='../data', train=True, download=True, transform=transform)
  test_dataset = CIFAR10(root='../data', train=False, download=True, transform=transform)
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)
  return train_loader, test_loader

def dataset_CIFAR100():
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
  train_dataset = CIFAR100(root='../data', train=True, download=True, transform=transform)
  test_dataset = CIFAR100(root='../data', train=False, download=True, transform=transform)
  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)
  return train_loader, test_loader



def Run(dataset, method):
    # Choose dataset
    if dataset=='MNIST': train_loader, test_loader = dataset_MNIST()
    elif dataset=='FMNIST': train_loader, test_loader = dataset_FMNIST()
    elif dataset=='CIFAR10': train_loader, test_loader = dataset_CIFAR10()
    elif dataset=='CIFAR100': train_loader, test_loader = dataset_CIFAR100()
    print(f'Finished loading {dataset} dataset')
    
    # Save log
    if exp_num!=-1:
        sys.stdout = open(f'./Result{exp_num}/{dataset}_{method}.txt', 'w')
        sys.stdout = open(f'./Result{exp_num}/{dataset}_{method}.txt', 'a')
    
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    for epoch in range(total_epoch):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data, epoch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (output.argmax(1) == target).sum().item()
            model.record_acc((output.argmax(1) == target).sum().item())
        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)
        train_acc_list.append(train_accuracy)
        train_loss_list.append(train_loss)
        
        # Test
        model.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data, epoch)
                loss = criterion(output, target)
                test_loss += loss.item()
                test_correct += (output.argmax(1) == target).sum().item()
            test_loss /= len(test_loader.dataset)
            test_accuracy = test_correct / len(test_loader.dataset)
            test_acc_list.append(test_accuracy)
            test_loss_list.append(test_loss)
            
        # Show Result
        print(f'Epoch {epoch + 1:2d} | '
            f'Train Loss: {train_loss:.4f} | Train Accuracy: {round(train_accuracy*100, 2)} | '
            f'Test Loss: {test_loss:.4f} | Test Accuracy: {round(test_accuracy*100, 2)}')
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list


# Plot
def plot(dataset, method, exp_num):
    figname = f"[{dataset}] {method}  #{exp_num}.svg"
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(train_acc_list, label='Train_acc')
    plt.plot(test_acc_list, label='Test_acc')
    plt.plot(train_loss_list, label='Train_loss')
    plt.plot(test_loss_list, label='Test_loss')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    if exp_num != -1 : plt.savefig(f'./Result{exp_num}/'+figname, format='svg')
    plt.show()


class SMD(nn.Module):
    def __init__(self, p=0.5, total_epoch=10, k=1):
        super(SMD, self).__init__()
        self.p = p
        self.total_epoch = total_epoch
        self.topk_masks = []
        self.mask = None
        self.accs = []
        self.k = k

    def forward(self, x, curr_epoch):
        if self.training:
            if curr_epoch <= (self.total_epoch/2):
                self.mask = (torch.rand_like(x) > self.p).float().to(device)
                x = x * self.mask / (1 - self.p)
                x = x.to(device)
                return x
            elif curr_epoch > (self.total_epoch/2):
                self.mask = random.choice(self.topk_masks)
                x = x * self.mask / (1 - self.p)
                return x
        if not self.training:
            return x

    def record_acc(self, acc):
        if len(self.topk_masks) < self.k:
            self.topk_masks.append(self.mask)
            self.accs.append(acc)
        elif len(self.topk_masks) >= self.k:    
            lowest = min(self.accs)
            if acc > lowest:
                lowest_idx = self.accs.index(lowest)
                self.topk_masks.pop(lowest_idx)
                self.accs.pop(lowest_idx)
                self.topk_masks.append(self.mask)
                self.accs.append(acc)

# MODELS ############################################################################################################


class Net_MNIST(nn.Module):
    def __init__(self):
        super(Net_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.fc1 = nn.Linear(4*4*128, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.smd1 = SMD(p=dropout_prob, total_epoch=total_epoch, k=top_k)
        self.smd2 = SMD(p=dropout_prob, total_epoch=total_epoch, k=top_k)
    def forward(self, x, curr_epoch):
        # 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        # 3
        x = x.view(-1, 4*4*128)
        x = F.relu(self.fc1(x))
        x = self.smd1(x, curr_epoch)
        x = F.relu(self.fc2(x))
        x = self.smd2(x, curr_epoch)
        x = self.fc3(x)
        return x
    def record_acc(self, acc):
        self.smd1.record_acc(acc)
        self.smd2.record_acc(acc)

# Initialize Model
model = Net_MNIST().to(device)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Train model
train_loss_list, train_acc_list, test_loss_list, test_acc_list = Run('MNIST', 'SMD')
plot('MNIST', 'SMD', exp_num)




class Net_FMNIST(nn.Module):
    def __init__(self):
        super(Net_FMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.fc1 = nn.Linear(4*4*128, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 100)
        self.smd1 = SMD(p=dropout_prob, total_epoch=total_epoch, k=top_k)
        self.smd2 = SMD(p=dropout_prob, total_epoch=total_epoch, k=top_k)

    def forward(self, x, curr_epoch):
        # 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        # 3
        x = x.view(-1, 4*4*128)
        x = F.relu(self.fc1(x))
        x = self.smd1(x, curr_epoch)
        x = F.relu(self.fc2(x))
        x = self.smd2(x, curr_epoch)
        x = self.fc3(x)
        return x

    def record_acc(self, acc):
        self.smd1.record_acc(acc)
        self.smd2.record_acc(acc)
        
# 모델 초기화
model = Net_FMNIST().to(device)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 모델 학습
train_loss_list, train_acc_list, test_loss_list, test_acc_list = Run('FMNIST', 'SMD')
plot('FMNIST', 'SMD', exp_num)




class Net_CIFAR10(nn.Module):
    def __init__(self):
        super(Net_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.fc1 = nn.Linear(5*5*128, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.smd1 = SMD(p=dropout_prob, total_epoch=total_epoch, k=top_k)
        self.smd2 = SMD(p=dropout_prob, total_epoch=total_epoch, k=top_k)

    def forward(self, x, curr_epoch):
        # 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        # 3
        x = x.view(-1, 5*5*128)
        x = F.relu(self.fc1(x))
        x = self.smd1(x, curr_epoch)
        x = F.relu(self.fc2(x))
        x = self.smd2(x, curr_epoch)
        x = self.fc3(x)
        return x
    def record_acc(self, acc):
        self.smd1.record_acc(acc)
        self.smd2.record_acc(acc)
        
# Initialize model
model = Net_CIFAR10().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Train model
train_loss_list, train_acc_list, test_loss_list, test_acc_list = Run('CIFAR10', 'SMD')
plot('CIFAR10', 'SMD', exp_num)





class Net_CIFAR100(nn.Module):
    def __init__(self):
        super(Net_CIFAR100, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.fc1 = nn.Linear(5*5*128, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 100)
        self.smd1 = SMD(p=dropout_prob, total_epoch=total_epoch, k=top_k)
        self.smd2 = SMD(p=dropout_prob, total_epoch=total_epoch, k=top_k)

    def forward(self, x, curr_epoch):
        # 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        # 3
        x = x.view(-1, 5*5*128)
        x = F.relu(self.fc1(x))
        x = self.smd1(x, curr_epoch)
        x = F.relu(self.fc2(x))
        x = self.smd2(x, curr_epoch)
        x = self.fc3(x)
        return x
    def record_acc(self, acc):
        self.smd1.record_acc(acc)
        self.smd2.record_acc(acc)
        
# Initialize model
model = Net_CIFAR100().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Train model
train_loss_list, train_acc_list, test_loss_list, test_acc_list = Run('CIFAR100', 'SMD')
plot('CIFAR100', 'SMD', exp_num)
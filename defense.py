import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def validation(model, testloader, device, T=4):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = F.log_softmax(outputs / T, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def fit(model, device, optimizer, scheduler, criterion, train_loader, val_loader, Temp=40, epochs=10):
    print("Fitting the model...")
    train_loss = []
    for epoch in range(epochs):
        loss_per_epoch = 0
        for data, label in tqdm(train_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            output = F.log_softmax(output/Temp, dim=1)
            # calculating loss on the output
            loss = criterion(output, label)
            optimizer.zero_grad()
            # grad calc w.r.t Loss func
            loss.backward()
            # update weights
            optimizer.step()
            loss_per_epoch += loss.item()
        print("Epoch: {} Loss: {} ".format(
            epoch+1, loss_per_epoch/len(train_loader)))
        train_loss.append(loss_per_epoch/len(train_loader))
        acc = validation(model, val_loader, device)
        print("Epoch: {} Accuracy: {} ".format(epoch+1, acc))
    return train_loss


def fit_distilled(teacher_model, distilled_model, device, optimizer, train_loader, val_loader, T=40, epochs=10):
    print("Fitting the distilled model")
    train_loss = []
    teacher_model.eval()
    distilled_model.train()
    for epoch in range(epochs):
        loss_per_epoch = 0
        for data, label in tqdm(train_loader):
            data, label = data.to(device), label.to(device)
            pseudo_targets = teacher_model(data)
            _q = F.softmax(pseudo_targets/T, dim=1)
            output = distilled_model(data)
            _p = F.log_softmax(output / T, dim=1)
            # calculating loss on the output
            loss = -torch.mean(torch.sum(_q * _p, dim=1))
            # grad calc w.r.t Loss func
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            loss_per_epoch += loss.item()
        print("Epoch: {} Loss: {} ".format(
            epoch+1, loss_per_epoch/len(train_loader)))
        train_loss.append(loss_per_epoch/len(train_loader))
        acc = validation(distilled_model, val_loader, device)
        print("Epoch: {} Accuracy: {} ".format(epoch+1, acc))
    return train_loss

# Testing fgsm

# # TODO : temperature  of distilled NN to define
# T=100
# inps = torch.zeros(196000*32, dtype=torch.float32).view(len(train_loader)*32,1,28,28)
# soft_lbl = torch.zeros(len(train_loader)*10*32, dtype=torch.float32,requires_grad =False).view(len(train_loader)*32,10)

# # Converting target labels to soft labels using modelF
# for i , (data,_) in tqdm(enumerate(train_loader.dataset)):
#     data, label = data.to(device), data.to(device)
#     data = data.reshape(1,1,28,28)
#     with torch.no_grad():
#         softlabel = teacher_model(data)
#         softlabel = F.softmax(softlabel, dim=1)
#         inps[i]=data
#         soft_lbl[i]=softlabel

# dataset = torch.utils.data.TensorDataset(inps,soft_lbl)
# train_loader_soft = torch.utils.data.DataLoader(dataset, batch_size=32, pin_memory=True)

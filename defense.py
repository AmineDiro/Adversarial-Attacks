import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from attack import fgsm_attack, random_fgsm_attack


def validation(model, testloader, device, T=4):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            outputs = F.log_softmax(outputs / T, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def fit(model, device, criterion, optimizer, train_loader, val_loader=None, T=1, epochs=10):
    train_loss =0
    correct =0
    total=0
    model.train()
    for epoch in range(epochs):
        for data, label in tqdm(train_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            # calculating loss on the output
            loss = criterion(output, label)
            optimizer.zero_grad()
            # grad calc w.r.t Loss func
            loss.backward()
            # update weights
            optimizer.step()
            final_pred = output.max(1, keepdim=True)[1].squeeze()
            correct += (final_pred == label).sum().item()
            train_loss += loss.item() * label.size(0)
            total += label.size(0)

        print('Epoch :{} Loss: {} Acc : {}'.format(
            epoch, train_loss/total, correct/total))
            
        if val_loader is not None : 
            acc = validation(model, val_loader, device)
            print("Epoch: {} Val accuracy: {} ".format(epoch+1, acc))  
        #Save after each epoch
        torch.save(model.state_dict(), "weights/base_training.pt")          
    return train_loss

# def fit(model, device, optimizer, scheduler, criterion, train_loader, val_loader, Temp=40, epochs=10):
#     print("Fitting the model...")
#     train_loss = []
#     for epoch in range(epochs):
#         loss_per_epoch = 0
#         for data, label in tqdm(train_loader):
#             data, label = data.to(device), label.to(device)
#             output = model(data)
#             output = F.log_softmax(output/Temp, dim=1)
#             # calculating loss on the output
#             loss = criterion(output, label)
#             optimizer.zero_grad()
#             # grad calc w.r.t Loss func
#             loss.backward()
#             # update weights
#             optimizer.step()
#             loss_per_epoch += loss.item()
#         print("Epoch: {} Loss: {} ".format(
#             epoch+1, loss_per_epoch/len(train_loader)))
#         train_loss.append(loss_per_epoch/len(train_loader))
#         acc = validation(model, val_loader, device)
#         print("Epoch: {} Accuracy: {} ".format(epoch+1, acc))
#     return train_loss


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


def adversarial_fit(model, device, optimizer, train_loader, val_loader=None, epsilon=0.3, alpha=0.3,  epochs=10):
    print("Fitting the model with adversarial training...")
    train_loss = 0
    correct = 0
    total = 0
    for epoch in range(epochs):
        loss_per_epoch = 0
        for data, label in tqdm(train_loader):
            X, y = data.to(device), label.to(device)
            # delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).to(device)
            # delta.requires_grad = True
            # output = model(X + delta)
            # loss = F.cross_entropy(output, y)
            # loss.backward()
            # grad = delta.grad.detach()
            # delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            # delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
            # delta = delta.detach()
            delta = fgsm_attack(model, X, y, epsilon)
            perturbed_data = torch.clamp(X + delta, 0, 1)
            output = model(perturbed_data)
            loss = F.cross_entropy(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_pred = output.max(1, keepdim=True)[1].squeeze()
            correct += (final_pred == y).sum().item()
            train_loss += loss.item() * y.size(0)
            total += y.size(0)
        print('Epoch :{} Loss: {} Acc : {}'.format(
            epoch, train_loss/total, correct/total))
    # TODO : Add val loss and val accuracy
    torch.save(model.state_dict(), "weights/adv_training.pt")
    return train_loss


def adversarial_fit_normal(model, device, optimizer, train_loader, val_loader=None, epsilon=0.3, alpha=0.5,  epochs=10):
    print("Fitting the model with adversarial training...")
    train_loss = 0
    correct = 0
    total = 0
    for epoch in range(epochs):
        loss_per_epoch = 0
        for data, label in tqdm(train_loader):
            X, y = data.to(device), label.to(device)
            perturbed_data = fgsm_attack(model, device, X, y, epsilon)
            output = model(perturbed_data)
            loss = (1-alpha)*F.cross_entropy(output, y) + \
                alpha*F.cross_entropy(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_pred = output.max(1, keepdim=True)[1].squeeze()
            correct += (final_pred == y).sum().item()
            train_loss += loss.item() * y.size(0)
            total += y.size(0)
        print('Epoch :{} Loss: {} Acc : {}'.format(
            epoch, train_loss/total, correct/total))
    # TODO : Add val loss and val accuracy
    torch.save(model.state_dict(), "weights/adv_training.pt")
    return train_loss


def adversarial_fit_random(model, device, optimizer, train_loader, val_loader=None, epsilon=0.3, alpha=0.5,  epochs=10):
    print("Fitting the model with adversarial training...")
    train_loss = 0
    correct = 0
    total = 0
    for epoch in range(epochs):
        loss_per_epoch = 0
        for data, label in tqdm(train_loader):
            X, y = data.to(device), label.to(device)
            perturbed_data = random_fgsm_attack(
                model, device, X, y, epsilon, alpha)
            output = model(perturbed_data)
            loss = (1-alpha)*F.cross_entropy(output, y) + \
                alpha*F.cross_entropy(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_pred = output.max(1, keepdim=True)[1].squeeze()
            correct += (final_pred == y).sum().item()
            train_loss += loss.item() * y.size(0)
            total += y.size(0)
        
        print('Epoch :{} Loss: {} Acc : {}'.format(
            epoch, train_loss/total, correct/total))
    # TODO : Add val loss and val accuracy
    return train_loss

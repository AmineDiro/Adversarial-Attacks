import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from Attacks.LBFGS import LBFGSAttack
from Attacks.FGSM import FGSMAttack
from Attacks.VanillaGradient import VanillaGradientAttack

from utils import * 


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
            loss = criterion(output/T, label)
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


def fit_distilled(teacher_model, distilled_model, device, optimizer, train_loader, val_loader= None, T=40, epochs=10):
    print("Fitting the distilled model ...")
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
        
        if val_loader is not None : 
           acc = validation(distilled_model, val_loader, device)
           print("Epoch: {} Val accuracy: {} ".format(epoch+1, acc))  
    return train_loss


###############  Adversarial training ############### 

def adversarial_fit_normal(model, device, optimizer, train_loader, val_loader=None, epsilon=0.3, alpha=0.5,  epochs=10):
    print("Training the model with adversarial training...")
    train_loss = 0
    correct = 0
    total = 0
    fgsm_attack = FGSMAttack(model, device, epsilon)
    for epoch in range(epochs):
        loss_per_epoch = 0
        for data, label in tqdm(train_loader):
            X, y = data.to(device), label.to(device)
            perturbed_data = fgsm_attack( X, y)
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
    print("Training the model with adversarial training...")
    train_loss = 0
    correct = 0
    total = 0
    for epoch in range(epochs):
        loss_per_epoch = 0
        for data, label in tqdm(train_loader):
            X, y = data.to(device), label.to(device)
            perturbed_data = random_fgsm_attack(
                model, device, X, y, epsilon, alpha,random=True)
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



###############  Binary Defense ############### 

def test_vanilla_binary_mnist(model, device, testloader, lmd, threshold=0.5):
    correct = 0
    total = 0
    adv_examples = []
    proba_distrib_adv = np.zeros(len(testloader))
    model.eval()
    vanilla_attack = VanillaGradientAttack(model, device,lmd=lmd)
    loop = tqdm(testloader, desc='Lambda = {}'.format(lmd))
    for i, (data, target) in enumerate(loop):
        # Generate target exemple where target is  different
        label_adv = torch.randint(0, 9, size=[target.size(0)])
        while (label_adv == target).sum().item() > 0:
            label_adv = torch.randint(0, 9, size=[target.size(0)])
        
        # Send targets img and label to device
        data , target  = data.to(device) , target.to(device)
        data.requires_grad = True
        # Get initial label
        data_binary = (data > threshold)*1
        # Binary segmentation of  original input
        output = model(data_binary.to(torch.float32))
        # Predict class of original input
        # index of the max log-probability
        init_pred = output.max(1, keepdim=True)[1]

        # Generate Adversarial exemple
        adv= vanilla_attack(data,target=label_adv)
        # Binary threashold adversary
        adv_binary = (adv > threshold)*1
        # Get model output of DNN
        output = model(adv_binary.to(torch.float32))
        final_pred = output.max(1, keepdim=True)[1].squeeze()
        # If same label changed
        # Get number of correctly classified examples
        correct += (final_pred == target).sum().item()
        total += target.size(0)
        if len(adv_examples) < 5:
            adversary = adv_binary.squeeze().detach().cpu().numpy()
            original = data.squeeze().detach().cpu().numpy()
            adv_examples.append(
                (original, adversary, init_pred, final_pred))
            # if i == 0 :
            #     imshow_adv_batch(model,data_binary,target,adv_binary.to(torch.float32))
    final_acc = correct / total
    print("Binary defense, Lambda {} Accuracy: {} ".format(lmd, final_acc))
    return final_acc, adv_examples 


def test_fgsm_mnist_binary(model, device, test_loader, epsilon, threshold=0.5):
    correct = 0
    adv_examples = []
    # Loop over  test set
    loop = tqdm(test_loader, desc='Iteration for epsilon = {}'.format(epsilon))
    fgsm_attack = FGSMAttack(model, device, epsilon)
    model.eval()
    for i, (data, target) in enumerate(loop):
        data, target = data.to(device), target.to(device)
        data_binary = (data > threshold)*1  #  Non-differentiable operation
        output = model(data_binary.to(torch.float32))
        # index of the max log-probability
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            # Skip this exemple
            continue

        #  FGSM Attack
        # Collect datagrad
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, target)
        # binary threshold data
        adv_binary = ((perturbed_data > threshold)*1).to(torch.float32)
        output = model(adv_binary)

        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():  # Nothing changed ( model has good defense)
            correct += 1
        else:
            #  Save au Max 5 adv exemples
            if len(adv_examples) < 5:
                adv_ex = adv_binary.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (init_pred.item(), final_pred.item(), adv_ex))
    final_acc = correct / len(test_loader)
    print("Binary defense, Epsilon {} Accuracy: {} ".format(epsilon, final_acc))
    return final_acc, adv_examples
    return train_loss

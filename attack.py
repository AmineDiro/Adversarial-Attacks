from __future__ import division
import time
import os
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import torchvision
from torchvision import datasets, models, transforms


############### Implementing FGSM ############### 

def fgsm_attack(image, epsilon, data_grad):
    #  Sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Add grad *epsilon to pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Clip to  [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def generate_adv_imagenet(model,device,testloader,epsilon):
    correct = 0
    adv_examples = []
    loop = tqdm(testloader)
    model.eval()
    for d, t in loop :
        data , target = d ,t 
        data.requires_grad = True
    
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] #index of the max log-probability
        
        if init_pred.item() != target.item():
            # Skip bad exemples in testdata
            continue            
        # Calculate negative log likelihood loss used
        loss = F.nll_loss(output, target) 
        model.zero_grad()
        # Backward pass
        loss.backward()
        
        ## FGSM Attack
        # Collect datagrad
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        
        # Getting the label
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item(): # Nothing changed ( model has good defense)
             correct += 1
        else :
            # Save exemple 
            if len(adv_examples) < 5:
                adversary = perturbed_data.squeeze().detach().cpu().permute(1, 2, 0)
                original = data.squeeze().detach().cpu().permute(1, 2, 0)
                adv_examples.append((original, adversary, init_pred , final_pred))
    final_acc = correct / float(len(testloader))
    return adv_examples, final_acc 

def test_fgsm_mnist(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []
    # Loop over  test set
    loop = tqdm(test_loader,desc='Iteration for epsilon = {}'.format(epsilon))
    
    for i, (data, target) in enumerate(loop):
        if i == 1000 :
            break
        # Utile ila kan 3anna GPU
        data, target = data.to(device), target.to(device)
        # requires_grad attribute of Data tensor.
        #  !/! Important for Attack
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] #index of the max log-probability

        if init_pred.item() != target.item():
            ## Skip this exemple
            continue

        # Calculate negative log likelihood loss used
        loss = F.nll_loss(output, target) 
        model.zero_grad()
        
        # Backward pass
        loss.backward()

        ## FGSM Attack
        # Collect datagrad
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Predict perturbed image class
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        if final_pred.item() == target.item(): # Nothing changed ( model has good defense)
            correct += 1
        else:
            #  Save au Max 5 adv exemples
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / 1000
    return final_acc, adv_examples

def test_fgsm_mnist_binary(model, device, test_loader, epsilon,threshold=0.5):
    correct = 0
    adv_examples = []
    # Loop over  test set
    loop = tqdm(test_loader,desc='Iteration for epsilon = {}'.format(epsilon))
    
    for i, (data, target) in enumerate(loop):
        if i == 1000 :
            break        
        data, target = data.to(device), target.to(device)
        # requires_grad attribute of Data tensor.
        #  !/! Important for Attack
        
        data.requires_grad = True
        data_binary = (data>threshold)*1 # Non-differentiable operation
        output = model(data_binary.to(torch.float32))
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] #index of the max log-probability
        if init_pred.item() != target.item():
            ## Skip this exemple
            continue
        # Calculate negative log likelihood loss used
        loss = F.nll_loss(output, target) 
        model.zero_grad()
        
        # Backward pass
        loss.backward()

        ## FGSM Attack
        # Collect datagrad
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # binary threshold data 
        adv_binary = ((perturbed_data>threshold)*1).to(torch.float32)
        output = model(adv_binary)
        
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item(): # Nothing changed ( model has good defense)
            correct += 1
        else:
            #  Save au Max 5 adv exemples
            if len(adv_examples) < 5:
                adv_ex = adv_binary.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    final_acc = correct / 1000
    return final_acc, adv_examples

###############  Implementing L-BFGS  ##################

def target_adversarial(model,x_target, device, n=0,epochs = 200,eta=0.5, lmd=0.05):
    """
   Function to generate an adversarial exemple based on a model, a target image and a label. 
   We need to have access to the gradient of the parameters in the model. 
    ...

    Parameters
    ----------
    model : torch model
    x_target : torch tensor
        the name of the animal
    n : int
        the sound that the animal makes
    device : torch.device
        the number of legs the animal has (default 4)
    epochs : int
        the number of epochs for gradient descent
    eta : float
        learning rate for gradient descent
    lmd : float 
        hyperparameter for 
    
    Returns
    -------
    x : torch tensor
        adversarial exemple
    """
    # Set the goal output
    goal = torch.tensor([n]).to(device)
    #
    x = torch.randn(x_target.size()).to(device)
    x.requires_grad = True
    
    # Gradient descent on the input
    for epoch in range(epochs):
        output = model(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, goal)
        model.zero_grad()
        # Backward pass
        loss.backward()
        # Get grad
        d = x.grad.data
        # The SGD update on x
        with torch.no_grad():
            # we don't need to update model params
            x -= eta * (d + lmd * (x - x_target)) 
            x.grad = None
    return x

def test_LBFGS_mnist(model,device,testloader,lmd):
    correct = 0
    adv_examples = []
    loop = tqdm(testloader,desc='Lambda = {}'.format(lmd))
    for i, (data, target) in enumerate(loop):
        if i == 1000 :
            break
        data = data.to(device)
        target = target.to(target)
        data.requires_grad = True
        
        # Get initial label
        output = model(data)          
        init_pred = output.max(1, keepdim=True)[1] #index of the max log-probability
        if init_pred.item() != target.item():
            # Skip bad exemples in testdata
            print('bad exemple')
            continue  
        n=torch.tensor(0)
        while n.item() == target.item():
            n = torch.randint(low=0,high=9,size=(1,))
        # Generate Adversarial exemple
        adv = target_adversarial(model,data,device,n,lmd=lmd)       
        output = model(adv)
        final_pred = output.max(1, keepdim=True)[1]
        # If same label changed
        if final_pred.item() == target.item(): 
             correct += 1
        else :
            if len(adv_examples) < 5:
                adversary = adv.squeeze().detach().cpu().numpy()
                original = data.squeeze().detach().cpu().numpy()
                adv_examples.append((original, adversary, init_pred , final_pred))
    final_acc = correct / 1000 # Normally float(len(testloader)) 
    return final_acc, adv_examples

def test_LBFGS_binary_mnist(model,device,testloader,lmd,threshold=0.5):
    correct = 0
    adv_examples = []
    loop = tqdm(testloader,desc='Lambda = {}'.format(lmd))
    for i, (data, target) in enumerate(loop):
        if i == 1000 :
            break
        data = data.to(device)
        target = target.to(target)
        data.requires_grad = True
        
        # Get initial label
        data_binary = (data>threshold)*1
        # Binary segment original input
        output = model(data_binary.to(torch.float32))
        # Predict class of original input
        init_pred = output.max(1, keepdim=True)[1] # index of the max log-probability
        if init_pred.item() != target.item():
            print('Bad exemple')
            continue  
        # Generate target exemple where target is  different
        label_adv=torch.tensor(0)
        while label_adv.item() == target.item():
            label_adv = torch.randint(low=0,high=9,size=(1,))
        # Generate Adversarial exemple
        adv = target_adversarial(model,data,device,n=label_adv,lmd=lmd)
        # Binary threashold adversary
        adv_binary = (adv>threshold)*1
        # Get model output of DNN
        output = model(adv_binary.to(torch.float32))
        final_pred = output.max(1, keepdim=True)[1]
        # If same label changed
        if final_pred.item() == target.item():             
            correct += 1
        else :
            if len(adv_examples) < 5:
                adversary = adv_binary.squeeze().detach().cpu().numpy()
                original = data.squeeze().detach().cpu().numpy()
                adv_examples.append((original, adversary, init_pred , final_pred))
    final_acc = correct / 1000
    return final_acc, adv_examples



def validation(model, testloader, device):
    correct = 0
    model.eval()
    for inputs, label in tqdm(testloader):
        inputs = inputs.to(device)
        output = model(inputs)
        final_pred = output.max(1, keepdim=True)[1] 
        if final_pred.item() == label.item(): 
             correct += 1 
    accuracy = correct/ float(len(testloader))
    return accuracy
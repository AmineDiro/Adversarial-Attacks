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

# Implementing 
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

def fgsm_attack(image, epsilon, data_grad):
    #  Sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Add grad *epsilon to pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Clip to  [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def generate_adv(model,testloader,epsilon):
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
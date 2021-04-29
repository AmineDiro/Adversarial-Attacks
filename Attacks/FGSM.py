import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import logging


class FGSMAttack():
    """
    Class FGSM attack. attacks by taking the direction to maximlize the linearized loss. 
    """

    def __init__(self, model, device, epsilon, random=False, alpha=None,T=1):
        self.model=model
        self.device=device
        self.epsilon=epsilon
        # Random start for FGSM
        self.random = random
        if self.random :
            if alpha is None :
                raise ValueError("Need an alpha value for random fgsm")
            self.alpha = alpha
        # To clamp between 0 and 1    
        self.min=0
        self.max=1
        #Temperature of softmax
        self.T =1

    def __call__(self, data, target):

        data = data.to(self.device)
        target = target.to(self.device)
        if not self.random:
            data_copy = data.detach().clone()
            data_copy.requires_grad = True
            output = self.model(data_copy)
            loss = F.cross_entropy(output/self.T, target)
            loss.backward()
            data_grad = data_copy.grad.detach()
            sign_data_grad = data_grad.sign()
            perturbed_image = data + self.epsilon*sign_data_grad
            perturbed_image = torch.clamp(perturbed_image, self.min, self.max)
            return perturbed_image
        else:
            rand_perturb = torch.FloatTensor(data.shape).uniform_(
                -self.epsilon, self.epsilon).to(self.device)
            x = data + rand_perturb
            x.clamp_(self.min, self.max)
            x.requires_grad = True
            self.model.eval()
            outputs = self.model(x)
            loss = F.cross_entropy(outputs, target)
            loss.backward()
            grad = x.grad.detach()
            x.data += self.alpha * torch.sign(grad.data)
            # Stay in L-inf of epsilon
            max_x = data + self.epsilon
            min_x = data - self.epsilon
            x = torch.max(torch.min(x, max_x), min_x)
            return x.clamp_(self.min, self.max)

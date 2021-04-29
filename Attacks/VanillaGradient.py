import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import logging


class VanillaGradientAttack():
    """
    Class Vanilla attack . Gradient descent on random noise to get close to the tagret image and label.
    """

    def __init__(self, model, device, epochs=100, eta=0.5, lmd=0.05, T=1, targeted=True):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.eta = eta
        self.lmd = lmd
        self.T = T #softmax temperature
        self.adv = None

    def __call__(self, x_target, target=None):
        x_target =x_target.to(self.device)    
        x = torch.randn(x_target.size()).to(self.device)
        if target is None:
            target = torch.randint(0, 9, size=[1]).to(self.device)
        target = target.to(self.device)
        # Back prop through random input
        x.requires_grad = True
        self.model.eval()
        # Gradient descent on the input
        for epoch in range(self.epochs):
            self.model.zero_grad()
            output = self.model(x)
            loss = F.cross_entropy(output/self.T, target)
            # Backward pass
            loss.backward()
            # Get grad
            d = x.grad.data
            # print('gradient',torch.norm(d))
            # The SGD update on x
            with torch.no_grad():
                # we don't need to update model params
                x -= self.eta * (d + self.lmd * (x - x_target))
                x.grad = None
        self.adv = x
        output = self.model(x)
        output = F.softmax(output/self.T, dim=1)
        self.output = output
        self.proba_adv = output.max().item()*100
        return  x
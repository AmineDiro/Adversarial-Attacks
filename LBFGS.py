import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import logging
from scipy.optimize import fmin_l_bfgs_b

class LBFGSAttack():
    """
    Class LBFGS attack L-BFGS-B to minimize the cross-entropy and the distance between the
    original and the adversary.
    """
    def __init__(self, model, device):
        self._adv = None  
        self.model = model
        self.device = device
        # Init bounds for lbfgs algo
        self.bounds = 0 , 1
        self._output = None

    def __call__(self, data,target, epsilon=0.01, steps=10):
        self.data = data.to(self.device)
        self.target = torch.tensor([target]).to(self.device) 
        # finding initial value for c        
        c = epsilon
        x0 = self.data.clone().cpu().numpy().flatten().astype(float)
        # Line search init
        for i in range(30):
            c = 2 * c
            # print('c={}'.format(c))
            is_adversary = self._lbfgsb(x0, c, steps)
            if is_adversary:
                # print('Successful')
                break
        if not is_adversary:
            # logging.info('Failed to init C ')
            return self._adv
        
        # binary search c
        c_low = 0
        c_high = c
        while c_high - c_low >= epsilon:
            c_half = (c_low + c_high) / 2
            is_adversary = self._lbfgsb(x0, c_half, steps)
            if is_adversary:
                c_high = c_high-epsilon
            else:
                c_low = c_half

    def _loss(self, adv_x, c):
        """
        Get the loss and gradient wr to adversary
        params:
        adv_x: the candidate adversarial example
        c: parameter 'C' in the paper        
        return: 
        (loss, gradient)
        """
        adv = torch.from_numpy(adv_x.reshape(self.data.size())).float().to(self.device).requires_grad_(True)
    
        # cross_entropy
        output = self.model(adv)
        ce = F.cross_entropy(output, self.target)
        # L2 distance
        d =  torch.sum((self.data - adv) ** 2)
        
        # Loss 
        loss = c * ce + d

        # gradient
        loss.backward()
        grad_ret = adv.grad.data.cpu().numpy().flatten().astype(float)
        loss = loss.data.cpu().numpy().flatten().astype(float)

        return loss, grad_ret

    def _lbfgsb(self, x0, c, maxiter):
        min_, max_ = self.bounds
        bounds = [(min_, max_)] * len(x0)
        approx_grad_eps = (max_ - min_) / 100.0
        x, f, d = fmin_l_bfgs_b(
            self._loss,
            x0,
            args=(c, ),
            bounds=bounds,
            maxiter=maxiter,
            epsilon=approx_grad_eps)
        if np.amax(x) > max_ or np.amin(x) < min_:
            x = np.clip(x, min_, max_)

        adv = torch.from_numpy(x.reshape(self.data.shape)).float().to(self.device)
        output = self.model(adv)
        adv_label = output.max(1, keepdim=True)[1]
        logging.info('target_label = {}, adv_label={}'.format(self.target, adv_label))
        # print('pre_label = {}, adv_label={}'.format(self.target, adv_label))
        self._adv = adv
        self._output = output
        return adv_label.item()== self.target.item()
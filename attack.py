from __future__ import division
import time
import os
import copy
from tqdm.auto import tqdm

import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Attacks.LBFGS import LBFGSAttack
from Attacks.FGSM import FGSMAttack
from Attacks.VanillaGradient import VanillaGradientAttack

from utils import *

############### Vanilla Attack  ##################


def test_vanilla_mnist(model, device, testloader, lmd,  T=1):
    print("####### Generating adversarial examples for Lambda ={} #######".format(lmd))
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
        data = data.to(device)
        label_adv = label_adv.to(device)
        target = target.to(device)

        output = model(data)
        # Predict class of original input
        # index of the max log-probability
        init_pred = output.max(1, keepdim=True)[1]

        ######### Generate Adversarial exemple #########
        # Fit vanilla attack on target image and target label
        vanilla_attack(data,target=label_adv)
        adv= vanilla_attack.adv
        #  get classes probabilities distib of adv prediction by model
        proba_adv = vanilla_attack.output
        proba_distrib_adv[i] = proba_adv.max().item()*100
        final_pred = proba_adv.max(1, keepdim=True)[1].squeeze()
        # Get number of correct exemples
        correct += (final_pred == target).sum().item()
        total += target.size(0)
        if len(adv_examples) < 5:
            adversary = adv.squeeze().detach().cpu().numpy()
            original = data.squeeze().detach().cpu().numpy()
            adv_examples.append(
                (original, adversary, init_pred, final_pred))
        # Print first exemple
        # if i ==0 :
        #     imshow_adv_batch(model,data,target,adv)
    final_acc = correct / total
    print("Lambda {} Accuracy: {} ".format(lmd, final_acc))
    return final_acc, adv_examples, proba_distrib_adv


############### Implementing FGSM ############### 

def test_fgsm_mnist(model, device, test_loader, epsilon,  T=1):
    correct = 0
    total = 0
    adv_examples = []
    proba_adv = np.zeros(len(test_loader))
    # Loop over  test set
    loop = tqdm(test_loader, desc='Iteration for epsilon = {}'.format(epsilon))
    model.eval()
    fgsm_attack = FGSMAttack(model, device, epsilon)
    for i, (data, target) in enumerate(loop):
        # GPU
        data, target = data.to(device), target.to(device)
        # requires_grad attribute of Data tensor.
        #  !/! Important for Attack
        output = model(data)
        # index of the max log-probability
        output = F.softmax(output / T, dim=1)
        init_pred = output.max(1, keepdim=True)[1]
        #  FGSM Attack : TODO : class FGSM attack
        # Collect datagrad
        perturbed_data = fgsm_attack(data, target)
        # Predict perturbed image class
        output = model(perturbed_data)
        # get the index of the max log-probability
        output = F.softmax(output/T, dim=1)
        proba_adv[i] = output.max().item()*100
        final_pred = output.max(1, keepdim=True)[1].squeeze()
        ## 
        correct += (final_pred == target).sum().item()
        total += target.size(0)
        if len(adv_examples) < 5:
            #adv_ex = perturbed_data[0].squeeze().detach().cpu().numpy()
            adversary = perturbed_data.squeeze().detach().cpu().numpy()
            original = data.squeeze().detach().cpu().numpy()
            adv_examples.append(
                (original, adversary, init_pred, final_pred))
            # adv_examples.append(
            #     (init_pred.cpu().numpy(), final_pred.cpu().numpy(), adv_ex))
        # if i ==0 :
        #     imshow_adv(model,data,target,perturbed_data)
    final_acc = correct / total

    print("Epsilon {} Accuracy: {} ".format(epsilon, final_acc))
    return final_acc, adv_examples, proba_adv


def test_fgsm_mnist_distilled(model, distilled_model, device, test_loader, epsilon, T=500):
    correct = 0
    adv_examples = []
    proba_adv = np.zeros(len(test_loader))
    # Loop over  test set
    loop = tqdm(test_loader, desc='Iteration for epsilon = {}'.format(epsilon))
    fgsm_attack = FGSMAttack(distilled_model, device, epsilon)
    for i, (data, target) in enumerate(loop):
        if i == len(test_loader):
            break
        # Utile ila kan 3anna GPU
        data, target = data.to(device), target.to(device)
        # requires_grad attribute of Data tensor.
        #  !/! Important for Attack
        data.requires_grad = True

        output = model(data)

        output = F.log_softmax(output / T, dim=1)
        # index of the max log-probability
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            # Skip this exemple
            continue

        # Calculate negative log likelihood loss used
        loss = F.nll_loss(output, target)
        model.zero_grad()

        # Backward pass
        loss.backward()

        #  FGSM Attack
        # Collect datagrad
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(distilled_model, target)

        # Predict perturbed image class
        output = distilled_model(perturbed_data)
        output = F.log_softmax(output / T, dim=1)
        # get the index of the max log-probability
        final_pred = output.max(1, keepdim=True)[1]
        proba_adv[i] = torch.exp(output).max().item()*100

        if final_pred.item() == target.item():  # Nothing changed ( model has good defense)
            correct += 1
        else:
            #  Save au Max 5 adv exemples
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / len(test_loader)
    print("Epsilon {} Accuracy: {} ".format(epsilon, final_acc))
    return final_acc, adv_examples, proba_adv


###############  Implementing L-BFGS  ##################

def test_LBFGS_mnist(model, device, testloader, nb_exemples=1000, T=1):
    correct = 0
    adv_examples = []
    proba_adv = np.zeros(nb_exemples)
    proba_orig = np.zeros(nb_exemples)
    loop = tqdm(testloader)
    lbfgs = LBFGSAttack(model, device)
    for i, (data, target) in enumerate(loop):
        if i == nb_exemples:
            break
        data = data.to(device)
        target = target.to(target)
        # Get initial label
        output = model(data)
        output = F.log_softmax(output / T, dim=1)
        # index of the max log-probability
        init_pred = output.max(1, keepdim=True)[1]
        proba_orig[i] = torch.exp(output).max().item()*100
        if init_pred.item() != target.item():
            # Skip bad exemples in testdata
            continue
        n = torch.tensor(0)
        while n.item() == target.item():
            n = torch.randint(low=0, high=9, size=(1,))

        # TODO : test Generate Adversarial exemple using LBFGS
        lbfgs(data, target=torch.randint(low=0, high=9, size=(1,)))
        adv = lbfgs._adv
        output = lbfgs._output
        proba_adv[i] = F.softmax(output / T, dim=1).max().item()*100
        final_pred = output.max(1, keepdim=True)[1]
        # Get probability
        # If same label changed
        if final_pred.item() == target.item():
            correct += 1
        else:
            if len(adv_examples) < 5:
                adversary = adv.squeeze().detach().cpu().numpy()
                original = data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (original, adversary, init_pred, final_pred))

    final_acc = correct / nb_exemples  # Normally float(len(testloader))
    return final_acc, adv_examples, proba_orig, proba_adv


############### BLACKBOX : CNN Imagenet test ###############
def generate_adv_imagenet(model, device, testloader, epsilon):
    correct = 0
    adv_examples = []
    loop = tqdm(testloader)
    model.eval()
    for d, t in loop:
        data, target = d, t
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            # Skip bad exemples in testdata
            continue
        # Calculate negative log likelihood loss used
        loss = F.nll_loss(output, target)
        model.zero_grad()
        # Backward pass
        loss.backward()

        #  FGSM Attack
        # Collect datagrad
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Getting the label
        output = model(perturbed_data)
        # get the index of the max log-probability
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():  # Nothing changed ( model has good defense)
            correct += 1
        else:
            # Save exemple 
            if len(adv_examples) < 5:
                adversary = perturbed_data.squeeze().detach().cpu().permute(1, 2, 0)
                original = data.squeeze().detach().cpu().permute(1, 2, 0)
                adv_examples.append(
                    (original, adversary, init_pred, final_pred))
    final_acc = correct / float(len(testloader))
    return adv_examples, final_acc

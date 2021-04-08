from __future__ import division
import time
import os
import copy
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from LBFGSAttack import LBFGSAttack


############### Vanilla Attack  ##################

def target_adversarial(model, x_target, device, n=0, epochs=100, eta=0.5, lmd=0.05, mode='log', T=1):
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
    #goal = torch.tensor([n]).to(device)
    #
    x = torch.randn(x_target.size()).to(device)
    x.requires_grad = True
    model.eval()
    # Gradient descent on the input
    for epoch in range(epochs):
        output = model(x)
        # No need for log softmax
        if mode != 'log':
            output = F.log_softmax(output/T, dim=1)
        loss = F.nll_loss(output, n)
        model.zero_grad()
        # Backward pass
        loss.backward()
        # Get grad
        d = x.grad.data
        # print('gradient',torch.norm(d))
        # The SGD update on x
        with torch.no_grad():
            # we don't need to update model params
            x -= eta * (d + lmd * (x - x_target))
            x.grad = None

    return x


def test_vanilla_mnist(model, device, testloader, lmd, mode="log", T=1):
    correct = 0
    total = 0
    adv_examples = []
    loop = tqdm(testloader, desc='Lambda = {}'.format(lmd))
    for i, (data, target) in enumerate(loop):
        if i == 1000:
            break
        data = data.to(device)
        # Generate target exemple where target is  different
        label_adv = torch.randint(0, 9, size=[target.size(0)])
        while (label_adv == target).sum().item() > 0:
            label_adv = torch.randint(0, 9, size=[target.size(0)])
        # Send targets to device
        label_adv = label_adv.to(device)
        target = target.to(device)

        data.requires_grad = True
        output = model(data)
        # Predict class of original input
        # index of the max log-probability
        init_pred = output.max(1, keepdim=True)[1]

        # Generate Adversarial exemple
        adv = target_adversarial(
            model, data, device, n=label_adv, lmd=lmd, mode=mode, T=T)
        # Get model output of DNN
        output = model(adv)
        if mode != 'log':
            # Output T = 1
            output = F.softmax(output, dim=1)
        final_pred = output.max(1, keepdim=True)[1].squeeze()
        # If same label changed
        correct += (final_pred == target).sum().item()
        total += target.size(0)
        if len(adv_examples) < 5:
            adversary = adv.squeeze().detach().cpu().numpy()
            original = data.squeeze().detach().cpu().numpy()
            adv_examples.append(
                (original, adversary, init_pred, final_pred))

    final_acc = correct / total
    print("Lambda {} Accuracy: {} ".format(lmd, final_acc))
    return final_acc, adv_examples

############### Implementing FGSM ############### 


# def fgsm_attack(image, epsilon, data_grad):
#     #  Sign of the data gradient
#     sign_data_grad = data_grad.sign()
#     # Add grad *epsilon to pixel of the input image
#     perturbed_image = image + epsilon*sign_data_grad
#     # Clip to  [0,1] range
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
#     return perturbed_image

# def normal_fgsm_attack(model, data, target, epsilon):
#     data_copy = data.detach().clone()
#     data_copy.requires_grad =True
#     output = model(data_copy)
#     loss = F.cross_entropy(output, target)
#     loss.backward()
#     data_grad = data_copy.grad.detach()
#     sign_data_grad = data_grad.sign()
#     perturbed_image = data + epsilon*sign_data_grad
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
#     return perturbed_image

def random_fgsm_attack(model,device, data, target, epsilon, alpha):
    rand_perturb = torch.FloatTensor(data.shape).uniform_(
                -epsilon, epsilon).to(device)
    x = data + rand_perturb
    x.clamp_(0,1)
    x.requires_grad = True
    model.eval()
    outputs = model(x)
    loss = F.cross_entropy(outputs, target)
    loss.backward()
    grad = x.grad.detach()
    x.data += alpha * torch.sign(grad.data) 
    # Stay in L-inf of epsilon
    max_x = data + epsilon
    min_x = data - epsilon
    x = torch.max(torch.min(x, max_x), min_x)
    return x.clamp_(0,1)

def fgsm_attack(model, device, data, target, epsilon):
    delta = torch.zeros_like(data, requires_grad=True).to(device)
    output = model(data + delta)
    loss = F.cross_entropy(output, target)
    loss.backward()
    grad = delta.grad.detach()
    delta.data = epsilon * torch.sign(grad)
    perturbed_data = torch.clamp(data + delta.detach(), 0, 1)
    return delta.detach()

def test_fgsm_mnist(model, device, test_loader, epsilon, mode='log', T=1):
    correct = 0
    total = 0
    adv_examples = []
    proba_adv = np.zeros(len(test_loader))
    # Loop over  test set
    loop = tqdm(test_loader, desc='Iteration for epsilon = {}'.format(epsilon))
    model.eval()
    for i, (data, target) in enumerate(loop):
        if i == len(test_loader):
            break
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
        perturbed_data = fgsm_attack(model, data, target, epsilon)
        # Predict perturbed image class
        output = model(perturbed_data)
        # get the index of the max log-probability
        output = F.softmax(output/T, dim=1)
        proba_adv[i] = output.max().item()*100
        final_pred = output.max(1, keepdim=True)[1].squeeze()

        # if final_pred.item() == target.item():  # Nothing changed ( model has good defense)
        correct += (final_pred == target).sum().item()
        total += target.size(0)
        if len(adv_examples) < 5:
            adv_ex = perturbed_data[0].squeeze().detach().cpu().numpy()
            adv_examples.append(
                (init_pred.numpy(), final_pred.numpy(), adv_ex))
    final_acc = correct / total
    print("Epsilon {} Accuracy: {} ".format(epsilon, final_acc))
    return final_acc, adv_examples, proba_adv


def test_fgsm_mnist_binary(model, device, test_loader, epsilon, threshold=0.5):
    correct = 0
    adv_examples = []
    # Loop over  test set
    loop = tqdm(test_loader, desc='Iteration for epsilon = {}'.format(epsilon))

    for i, (data, target) in enumerate(loop):
        if i == 1000:
            break
        data, target = data.to(device), target.to(device)
        # requires_grad attribute of Data tensor.
        #  !/! Important for Attack

        data.requires_grad = True
        data_binary = (data > threshold)*1  #  Non-differentiable operation
        output = model(data_binary.to(torch.float32))
        output = model(data)
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
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
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
    final_acc = correct / 1000
    return final_acc, adv_examples


def test_fgsm_mnist_distilled(model, distilled_model, device, test_loader, epsilon, mode='log', T=100):
    correct = 0
    adv_examples = []
    proba_adv = np.zeros(len(test_loader))
    # Loop over  test set
    loop = tqdm(test_loader, desc='Iteration for epsilon = {}'.format(epsilon))

    for i, (data, target) in enumerate(loop):
        if i == len(test_loader):
            break
        # Utile ila kan 3anna GPU
        data, target = data.to(device), target.to(device)
        # requires_grad attribute of Data tensor.
        #  !/! Important for Attack
        data.requires_grad = True

        output = model(data)

        if mode != 'log':
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
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

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


#  CNN Imagenet test
def generate_adv_imagenet(model, device, testloader, epsilon):
    correct = 0
    adv_examples = []
    loop = tqdm(testloader)
    model.eval()
    for d, t in loop:
        data, target = d, t
        data.requires_grad = True

        output = model(data)
        # index of the max log-probability
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


###############  Implementing L-BFGS  ##################

def test_LBFGS_mnist(model, device, testloader, nb_exemples=1000, T=1, mode='log'):
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
        if mode != 'log':
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

        # Generate Adversarial exemple using LBFGS
        lbfgs._apply(data, target=0)
        adv = lbfgs._adv
        output = lbfgs._output
        # Get prediction
        if mode != 'log':
            output = F.log_softmax(output / T, dim=1)

        final_pred = output.max(1, keepdim=True)[1]
        # Get probability
        proba_adv[i] = torch.exp(output).max().item()*100
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


def test_LBFGS_binary_mnist(model, device, testloader, lmd, threshold=0.5):
    correct = 0
    adv_examples = []
    loop = tqdm(testloader, desc='Lambda = {}'.format(lmd))
    for i, (data, target) in enumerate(loop):
        if i == 1000:
            break
        data = data.to(device)
        target = target.to(target)
        data.requires_grad = True

        # Get initial label
        data_binary = (data > threshold)*1
        # Binary segment original input
        output = model(data_binary.to(torch.float32))
        # Predict class of original input
        # index of the max log-probability
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            print('Bad exemple')
            continue
        # Generate target exemple where target is  different
        label_adv = torch.tensor(0)
        while label_adv.item() == target.item():
            label_adv = torch.randint(low=0, high=9, size=(1,))
        # Generate Adversarial exemple
        adv = target_adversarial(model, data, device, n=label_adv, lmd=lmd)
        # Binary threashold adversary
        adv_binary = (adv > threshold)*1
        # Get model output of DNN
        output = model(adv_binary.to(torch.float32))
        final_pred = output.max(1, keepdim=True)[1]
        # If same label changed
        if final_pred.item() == target.item():
            correct += 1
        else:
            if len(adv_examples) < 5:
                adversary = adv_binary.squeeze().detach().cpu().numpy()
                original = data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (original, adversary, init_pred, final_pred))
    final_acc = correct / 1000
    return final_acc, adv_examples


#  Utils
def validation(model, testloader, device):
    correct = 0
    model.eval()
    for inputs, label in tqdm(testloader):
        inputs = inputs.to(device)
        output = model(inputs)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == label.item():
            correct += 1
    accuracy = correct / float(len(testloader))
    return accuracy

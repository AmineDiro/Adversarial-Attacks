import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def imshow_adv(model, data, target, adv, T=1):
    output = model(adv)
    output = F.softmax(output/T, dim=1)
    proba_adv = output.max()*100
    final_pred = output.max(1, keepdim=True)[1].squeeze()
    # To cpu then to numpy
    adv_numpy = adv.squeeze().detach().cpu().numpy()
    data_numpy = data.squeeze().detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    final_pred = final_pred.detach().cpu().numpy()
    proba_adv = proba_adv.detach().cpu().numpy()
    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].imshow(data_numpy, cmap='gray')
    ax[0].set_title("Original Label {}".format(target))
    ax[1].imshow(adv_numpy, cmap='gray')
    ax[1].set_title("Adversarial Label {}, with : {:.2f}% probability".format(
        final_pred, proba_adv))
    plt.show()
    return


def imshow_adv_batch(model, data, target, adv):
    output = model(adv)
    final_pred = output.max(1, keepdim=True)[1].squeeze()
    adv_numpy = adv[0].squeeze().detach().cpu().numpy()
    data_numpy = data[0].squeeze().detach().cpu().numpy()
    target = target.detach().cpu()
    final_pred = final_pred.detach().cpu()
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].imshow(data_numpy, cmap='gray')
    ax[0].set_title("Original Label {}".format(target[0].item()))
    ax[1].imshow(adv_numpy, cmap='gray')
    ax[1].set_title("Adversarial Label {}".format(final_pred[0].item()))
    plt.show()

def validation(model, testloader, device, T=1):
    correct = 0
    total = 0 
    model.eval()
    for inputs, label in tqdm(testloader, desc='Accuracy for testset'):
        inputs = inputs.to(device)
        label = label.to(device)
        output = model(inputs)
        final_pred = output.max(1, keepdim=True)[1].squeeze()
        correct += (final_pred == label).sum().item()
        total += label.size(0)
    accuracy = correct / total
    return accuracy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def imshow_adv(model,data,target,adv):
    output = model(adv)
    final_pred = output.max(1, keepdim=True)[1].squeeze()
    adv_numpy = adv[0].squeeze().detach().cpu().numpy()
    data_numpy = data[0].squeeze().detach().cpu().numpy()
    target = target.detach().cpu()
    fig, ax = plt.subplots(1,2,figsize=(12,8))
    ax[0].imshow(data_numpy,cmap='gray')
    ax[0].set_title("Original Label {}".format(target[0].item()))
    ax[1].imshow(adv_numpy,cmap='gray')
    ax[1].set_title("ADversarial Label {}".format(final_pred[0].item()))
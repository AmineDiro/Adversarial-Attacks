import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from LBFGS import LBFGSAttack
from FGSM import FGSMAttack
from VanillaGradient import VanillaGradientAttack


class Tester():
    """
    Run experiments of attacks
    """

    def __init__(self, model, device, dataset, bs=1, T=1):
        self.model = model
        self.device = device
        self.T = T
        self.testloader = torch.utils.data.DataLoader(
            dataset, batch_size=bs, shuffle=True)

    def test_attack(self, attack, params):
        correct = 0
        total = 0
        adv_examples = []
        proba_adv = np.zeros(len(test_loader))
        # Loop over  test set
        loop = tqdm(
            test_loader, desc='Iteration for epsilon = {}'.format(epsilon))
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
            perturbed_data = fgsm_attack(model, device, data, target, epsilon)
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
                    (init_pred.cpu().numpy(), final_pred.cpu().numpy(), adv_ex))
        final_acc = correct / total
        print("Epsilon {} Accuracy: {} ".format(epsilon, final_acc))
        return final_acc, adv_examples, proba_adv

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U9TYyaSjs1hsp3p-C7Y2zNUhHHHzgd4p) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)


# Adversarial examples

Implementation of adversarial attack on different deep NN classifiers, the attacks are base on the algorithms in the papers :

**ATTACKS :**
* LBFGS Attack : [Explaining and harnessing adversarial examples](https://arxiv.org/pdf/1412.6572v3.pdf).
* FGSM : [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199).
*  [Adversarial Attacks and Defences Competition](https://arxiv.org/pdf/1804.00097.pdf).


**DEFENSES :**
* Distilled neural network : [Distillation as a Defense to Adversarial
Perturbations against Deep Neural Networks](https://arxiv.org/pdf/1511.04508.pdf)
* FGSM Training ?
* 

 
## Paper structure
- Introduction to adversarial exemples : Robustness of neural networks
- Explain white vs black box attacks
    * white box: Atatcker has full access to model, architecture, inputs, outputs, and weights
    * black box : Attacker only has access to the inputs and outputs of the model
- Implementation  :  defenses vs attacks
    * Attacks : 
        * Vanilla SGD attack
        * LBFGS attack
        * FGSM Attack
        * ATN ?? 
        * Get exemples and accuracy and mean proba results
    * Defense : 
        * Binary threashold : results
        * Neural network distillation
- Tests : 
    - Generalization across different models
    - Robustess of attacks and defenses
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
    - Understanding Adversarial Examples : An adversarial example is an instance with small, intentional feature perturbations that cause a machine learning model to make a false prediction. Adversarial examples are counterfactual examples with the aim to deceive the model.
    - From FGSM paper Adversarial examples aren’t a product of overfitting, but rather of high-dimensional input and the internal linearity of modern models => FGSM paper
    - Manifold hypothesis : 
        * there is only a relatively small region of a very high dimensional space in which inputs exist
        * property of adversarial examples is transferability :  adversarial examples generated for one model will typically work on another. 

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

## TODO : 
* Refactor:
    - Vanillagradient into class
    - Tester class for attacks
    
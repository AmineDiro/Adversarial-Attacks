[![Adversarial Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z5b1yvpLm7zaBK0Oz4otqQ3zWDyPlM42?usp=sharing) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)



# Adversarial examples

Implementation of adversarial attack on different deep NN classifiers, the attacks are base on the algorithms in the papers :

**ATTACKS :**
* LBFGS Attack : [Explaining and harnessing adversarial examples](https://arxiv.org/pdf/1412.6572v3.pdf).
* FGSM : [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199).
* Vanilla Attack :  



**DEFENSES :**
* Distilled neural network : [Distillation as a Defense to Adversarial
Perturbations against Deep Neural Networks](https://arxiv.org/pdf/1511.04508.pdf)
* Adversarial Training
* Binary thresholding 

 
Project Organization
-----------------------

    ├── Adversarial_blackbox_attacks.ipynb  <- Notebook for testing BB attacks>
    ├── Adversarial_whitebox_attacks.ipynb  <- Notebook for testing WB attacks>
    ├── attack.py                           <- Test functions of attacks>
    ├── Attacks                             
    │   ├── FGSM.py                         <- FGSM fast attack class>
    │   ├── LBFGS.py                        <- L-BFGS attack class >
    │   └── VanillaGradient.py              <- Vanilla attack class >      
    ├── Defense.ipynb                       <- Notebook for testing defenses>
    ├── defense.py                          <- Test functions of defense>
    ├── imagenet_classes.txt
    ├── Net.py                              <- Architectures of models >
    ├── Results                             <- Resulting images and accuracies >
    ├── utils.py                            <- Plotting functions >
    └── weights                             <- Weights for pretrained models>




Our work was inspired by [Adversarial Attacks and Defences Competition](https://arxiv.org/pdf/1804.00097.pdf), we implemented 3 differents attack vectors and 3 matching defenses.  

- `Adversarial_whitebox_attacks.ipynb`  : We first implemented the attacks on the architecture `Net.py` with MNIST dataset, the notebook show the impact of our different attacks  on the accuracy of the model
    - [![Adversarial_whitebox_attacks](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1J86NJTDyyAZq4Y6zR7__n5ACw4OMWAdf?usp=sharing) 
- `Defense.ipynb` : This notebook showcases the robustness of 3 different defenses against the attacks. You'll find the accuracy measure of the model when adding the defense. The `L-BFGS` attack was left out of the testing because the high computationnal demand of the attack.
    - [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CVC08o4i0GKCbZzjH5MexZKsNp_Vd8mD?usp=sharing)
- `Adversarial_blackbox_attacks.ipynb` :  One very interesting feature of adversarial examples is their ability to attack different models. We tested this unique property by attack a model based on image generated froma  different one. We used a more complex dataset (ants/bees) of 3 channels images  from this test. 
    - [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z5b1yvpLm7zaBK0Oz4otqQ3zWDyPlM42?usp=sharing) 
    - **Note** the dataset is available to download by running :
        ```bash 
            wget https://download.pytorch.org/tutorial/hymenoptera_data.zip
        ```
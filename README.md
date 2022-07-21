# Image Classification using Multiple Convolutional Neural Networks on the Fashion-MNIST dataset

It is a code for classification of the Fashion-MNIST dataset using Multiple Convolutional Neural Networks (MCNN). The parameters (convolutional layers, learning rate, etc.) are turned using Ray turn (https://docs.ray.io/en/master/tune/examples/tune-pytorch-cifar.html). The performance of different MCNN models are compared to state of the models (alexnet[1], lenet[2], effcientnet[3], mobilenet[4], resnet[5], vit[6]).

## Prerequisites
* Ubuntu 16.04

* Python 3

* pytorch

## run
Alexnet 
```
python3 alexnet_fashion_mnist.py
```
Efficientnet
```
python3 efficientnet_fashion_mnist.py
```
Lenet 
```
python3 lenet_fashion_mnist.py
```
Mobilenet
```
python3 mobilenet_fashion_mnist.py
```
Resnet
```
python3 resnet18_fashion_mnist.py
```
VIT
```
python3 VIT_fashion_mnist.py
```
MCNN15 (Our proposed network model)
```
python3 MCNN9_fashion_mnist_W_ray.py
python3 MCNN12_fashion_mnist_W_ray.py
python3 MCNN15_fashion_mnist_W_ray.py
```
# Image Classification using Multiple Convolutional Neural Networks on the Fashion-MNIST dataset
As the number of elderly people grows, service robots or robotic arms, which can accomplish complex tasks such as dressing disabled people, are increasingly demanded. Consequently, there is a growing interest in studying dressing tasks and, to dress a person, the detection and classification of household objects are taken into consideration. For this reason, we proposed to study in this paper image classification with four different neural network models to improve image classification accuracy on the Fashion-MNIST dataset. The network models with the highest accuracy are tested with the Fashion Product and with a customized dataset. The results show that one of our models, Multiple Convolutional Neural Networks included 15 Convolutional layers (MCNN15), boosted the state of art accuracy, obtaining a classification accuracy of 94.09\% on the Fashion-MNIST dataset with respect to the literature. Moreover, MCNN15 with the Fashion Product dataset and the household dataset obtains 60\% and 40\% of accuracy respectively.

This work is submitted to Journal paper.

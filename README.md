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

@article{nocentini2022image,
  title={Image Classification Using Multiple Convolutional Neural Networks on the Fashion-MNIST Dataset},
  author={Nocentini, Olivia and Kim, Jaeseok and Bashir, Muhammad Zain and Cavallo, Filippo},
  journal={Sensors},
  volume={22},
  number={23},
  pages={9544},
  year={2022},
  publisher={MDPI}
}

###  References
1. Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems 25 (2012).
2. LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.
3. Tan, Mingxing, and Quoc Le. "Efficientnet: Rethinking model scaling for convolutional neural networks." International conference on machine learning. PMLR, 2019.
4. Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." arXiv preprint arXiv:1704.04861 (2017).
5. He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
6. Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).

# FashionMNIST Classification with LeNet-5 (PyTorch)

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) based on the **LeNet-5 architecture** to classify images from the **FashionMNIST** dataset.

Unlike standard MNIST (digits), FashionMNIST consists of 28x28 grayscale images of 10 fashion categories, presenting a slightly more challenging computer vision task.

## Dataset Classes
The model classifies images into the following categories:
`T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`.

## Model & Architecture
* **Input:** 1-Channel Grayscale Images (28x28)
* **Architecture:** LeNet-5 (modified for 1 channel input)
* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam (Learning Rate: 1e-4)

##  Usage

Install dependencies:
```bash
pip install torch torchvision matplotlib tqdm numpy

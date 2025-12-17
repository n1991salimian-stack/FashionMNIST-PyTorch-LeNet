"""
Project: FashionMNIST Classification using LeNet-5
Framework: PyTorch
Description: Classifying clothing articles (Grayscale images) using CNN.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

print(f"CUDA Available: {torch.cuda.is_available()}")

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # in_channels=1 because FashionMNIST is grayscale
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.c2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.c3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, img):
        x = self.tanh(self.c1(img))
        x = self.tanh(self.avg_pool(x))
        x = self.tanh(self.c2(x))
        x = self.tanh(self.avg_pool(x))
        x = self.tanh(self.c3(x))
        x = torch.flatten(x, 1)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

def get_dataset(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize for grayscale
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                 download=True, transform=transform)

    # Split dataset into train and validation set
    trainset, valset = torch.utils.data.random_split(trainset, [57500, 2500])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)

    return trainloader, valloader, testloader

class Model:
    def __init__(self, model, learning_rate, device):
        self.model = model
        self.lr = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.device = device

    def batch_accuracy(self, output, target):
        output = nn.functional.softmax(output, dim=1)
        output = output.argmax(1)
        acc = torch.sum(output == target) / output.shape[0]
        return acc.cpu().item() * 100

    def train_step(self, dataset):
        self.model.train()
        batch_loss = []
        batch_acc = []
        for batch in dataset:
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            self.opt.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss(outputs, targets)
            loss.backward()
            self.opt.step()
            batch_loss.append(loss.item())
            batch_acc.append(self.batch_accuracy(outputs, targets))

        self.train_loss.append(np.mean(batch_loss))
        self.train_acc.append(np.mean(batch_acc))

    def validation_step(self, dataset):
        self.model.eval()
        batch_loss = []
        batch_acc = []
        with torch.no_grad():
            for batch in dataset:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)

                outputs = self.model(inputs)

                loss = self.loss(outputs, targets)
                batch_loss.append(loss.item())
                batch_acc.append(self.batch_accuracy(outputs, targets))

        self.val_loss.append(np.mean(batch_loss))
        self.val_acc.append(np.mean(batch_acc))

    def test_step(self, dataset):
        self.model.eval()
        batch_acc = []
        with torch.no_grad():
            for batch in dataset:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)

                outputs = self.model(inputs)
                batch_acc.append(self.batch_accuracy(outputs, targets))

        print(f"Final Test Set Accuracy: {np.mean(batch_acc):.2f}%")


if __name__ == "__main__":
    epochs = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    
    learning_rate = 1e-4
    batch_size = 32
    
    lenet5 = LeNet5().to(device)
    train_loader, val_loader, test_loader = get_dataset(batch_size)
    
    model = Model(lenet5, learning_rate, device)
    
    # Progress bar loop
    for epoch in tqdm(range(epochs), desc='Epoch'):
        model.train_step(train_loader)
        model.validation_step(val_loader)
    
    # Evaluate on test set
    model.test_step(test_loader)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(model.train_acc, label='Train Accuracy')
    plt.plot(model.val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('FashionMNIST Training Performance')
    plt.show()

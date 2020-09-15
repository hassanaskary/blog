---
layout: post
toc: true
title: Getting Started with PyTorch
description: Guide to understanding PyTorch and developing your first model using it.
categories: [python, pytorch]
comments: false
---

PyTorch is a machine learning framework by Facebook. It allows you to do tensor computation on the GPU, and design, train, and deploy deep learning models.

In this post we'll cover the following topics:

1. Installing PyTorch
2. Getting familiar with commonly used APIs
3. Building a classifier for MNIST

I will assume familiarity with machine learning and deep learning.

# Installing PyTorch

Go to the official [website](https://pytorch.org/get-started/locally/) to download and install PyTorch. Scroll down a bit and you'll see the "Start Locally" section. Select the option according to your OS, CUDA version, and preference of Pip or Conda.

I'm using Pop OS 20.04 right now without GPU. So I'll select the following:

![]({{ site.baseurl }}/images/getting-started-with-pytorch/pytorch-get-started-locally.png "PyTorch get started locally page.")

Copy and paste the command indicated by "Run this Command" into your terminal and execute. This should install PyTorch on your system.

# Getting Familiar with Commonly Used APIs

PyTorch consists of many APIs. I won't cover individual syntax and commands but give you an overview of what each API does. This way you will have a birds eye view of PyTorch and when you get stuck you will know where to look.

The commonly used APIs/libraries are:

1. torch
    1. torch.nn
    2. torch.nn.functional
    3. torch.optim
    4. torch.utils.data
1. torchvision

## torch

The torch API provides the tensor data structure and its associated methods and mathematical operations. A tensor is similar to a numpy array. Difference between a numpy array and a torch tensor is that the latter can utilize a GPU to do computations.

### torch.nn

The torch.nn API contains the building blocks to design and build a neural network. These building blocks are called modules and they are subclasses that inherit from `torch.nn.Module` base class. They are stateful meaning they automatically store their weights, gradients and other attributes. They also need to be initialized before usage.

The API contains fully-connected linear layers, convolution layers, activation functions, loss functions, normalization layers, dropout etc.

### torch.nn.functional

The torch.nn.functional API contains useful functions like activations and loss functions etc. However, they are not stateful meaning they won't store their attributes automatically. Instead we will have to store them manually.

These functions can be used for applying layers that don't need to be learned like activations or pooling etc.

This API provides additional flexibility in designing your models.

### torch.optim

The torch.optim API contains optimizers like Adam, SGD, and RMSprop etc. and methods for calculating gradients and applying them.

### torch.utils.data

This is a utility that provides the `DataLoader` class which is used to load data from a folder or from a built-in dataset.

## torchvision

The torchvision library contains datasets, models, and utilities related to computer vision. The `torchvision.datasets` package contains popular datasets like MNIST, ImageNet, and COCO etc. The `torchvision.models` package contains famous pre-trained or untrained vision models like ResNet, Inception, and Mask R-CNN etc. The `torchvision.transforms` package contains image transformations used to preprocess datasets like converting images to torch tensor, cropping, or resizing etc.

# Building a Classifier for MNIST

There are many ways to build a neural network in PyTorch. In this post I'll demonstrate building a convolutional neural network. It will have three convolution layers followed by three linear layers with Relu activations and finally a cross entropy loss.

We will be doing things in the following order:

1. Import packages
2. Load and preprocess data
3. Build the network
4. Write the training loop
5. Write the testing loop


## Import Packages

First we will import the required packages.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
```

## Load and Preprocess Data

We will use the built-in MNIST digits dataset. But first we have to specify how we will preprocess the images.

```python
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])
```

Here, `transforms.Compose()` is a list of the transformations we want to apply to the images in order. First, we will convert the images to torch tensors then we will normalize them.

The syntax of `transforms.Normalize()` is as follows:

```python
transforms.Normalize(mean, standard deviation)
```

Each index in those two tuples correspond to a channel in the image. Since MNIST images are grayscale they have only one channel. So we are setting the mean and standard deviation of that one channel to 0.5 yielding a mean of 0 and standard deviation of 1.

Lets download the dataset.

```python
trainset = datasets.MNIST('dataset/', train=True, download=True, transform=preprocess)
testset = datasets.MNIST('dataset/', train=False, download=True, transform=preprocess)
```

Here, we are creating the train and test sets. They will be downloaded in "dataset/" folder. The `train` argument specifies whether to download train or test set. The `transform` argument takes in the `transforms.Compose()` object and preprocesses the images according to it.

Now we will create data loaders that will return a batch of data on each iteration.

```python
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
```

## Build the Network

Finally the fun part. First we will define some constants.

```python
nf = 32
lr = 0.0001
beta1 = 0.5
beta2 = 0.999
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Here, `nf` is the number of kernels/filters/neurons (from now on I will call them filters). For `device` we are checking whether a CUDA enabled GPU exists. If it does not exist then we will use the CPU.

Now lets define the network. In PyTorch by convention we define models as a subclass of `torch.nn.Module`. We initialize layers in `__init__()` and connect them in `forward()` function.

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # resultant size will be 32x14x14

            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # resultant size will be 32x7x7
        )

        self.linears = nn.Sequential(
            nn.Linear(1568, 100), # width=7, height=7, filters=32; linear layer input = 7*7*32 = 1568
            nn.ReLU(),

            nn.Linear(100, 50),
            nn.ReLU(),

            nn.Linear(50, 10),
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1) # flattening, result will be (64, 1568)
        x = self.linears(x)
        return x
```

Ok, a lot to unpack here. In the `__init__()` you can see `self.convs` and `self.linears` being defined. The `nn.Sequential` is like a list. It contains modules and executes them in order sequentially.

We are defining the convolution part and the fully-connected or linear part separately. All layers will be initialized randomly.

Now, lets look at the individual layers. In `self.convs` we first see `nn.Conv2d()` this is a convolution layer. The arguments correspond to:

```
nn.Conv2d(number of input channels, number of filters, filter size, stride, padding)
```

This convolution layer is connected to a relu layer. This continues until we reach a max pool layer. Its arguments correspond to:

```
nn.MaxPool2d(filter size, stride)
```

Moving on in `self.linears` we have the fully connected layers. Its arguments correspond to:

```
nn.Linear(input feature/vector size, number of neurons in layer)
```

In the first linear layer we see that the input is 1568 and the output or number of neurons is 100. In PyTorch we have to define and initialize our layers before using them. This means we have to specify there input and output sizes.

I have calculated the output size of `self.convs`. We can calculate the output size of convolution using the following formulas:

```
outputWidth = (inputWidth - filterSize + (2 * padding)) / stride + 1

outputHeight = (inputHeight - filterSize + (2 * padding)) / stride + 1
```

There are also max pool layers which change the size of our features. We can calculate the output size of max pool using:

```
outputWidth = (inputWidth - filterSize) / stride + 1

outputHeight = (inputHeight - filterSize) / stride + 1
```

This calculation can be automated by writing a custom module. But that is out of the scope of this guide.

Moving on to the `forward()` function. This is where we bring together the `self.convs` and `self.linears` and connect them. Here `x` is the input features/tensors.

After getting output from `self.convs` we flatten it by using the `view()` function. The `view()` function is similar to `reshape()` in numpy. Here the first argument is set to `x.size(0)` which is the batch size. The second argument is "-1" which tells PyTorch to infer the remaining size which in this case will be equal to 1568. After this the tensor will be flattened and ready to be passed to `self.linears`. Finally, we will return the result.

## Write the Training Loop

Lets initialize the model. Define the loss function and optimizer.

```python
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
criterion_CE = nn.CrossEntropyLoss().to(device)
```

Here the `.to(device)` means to transfer the model and loss function to the specified device. If a CUDA enabled GPU was found on the system then the model and loss will utilize it.

Now lets write the training loop.

```python
epoch = 10
model.train()
for e in range(epoch):
    print(f'Starting epoch {e} of {epoch}')
    for X, y in trainloader:
        X = X.to(device)
        predictions = model(X)
        optimizer.zero_grad()
        loss = criterion_CE(predictions, y)
        loss.backward()
        optimizer.step()
    print(f'Loss: {loss.item()}')

torch.save(model.state_dict(), "model.pt")
```

We will train for 10 epochs. Before training we set the model to training mode since behaviour of some layers like Batchnorm and Dropout is different in eval mode and train mode. In this case we are not using these kind of layers but it's best practice to set the mode.

We iterate over the trainloader which return a batch of images and labels for each iteration. First we transfer the images to specified device then we pass the images to our model and get predictions. Then we zero out our gradients and calculate the loss. After this we calculate our gradient with `loss.backward()` and finally apply those gradients with `optimizer.step()`. That's it we loop over these steps for the whole dataset.

At the end we save the weights of the trained model. The training should take at most 30 minutes without GPU.

## Write the Testing Loop

Lets test our model.

```python
model.eval()
correct = 0
for X, y in testloader:
    with torch.no_grad():
        X = X.to(device)
        output = model(X)
        predictions = output.max(1)[1]
        correct += torch.eq(predictions, y).sum()

print(f'accuracy: {int(correct)}/{len(testloader.dataset)} ({int(correct)/len(testloader.dataset)} or {int(correct)/len(testloader.dataset) * 100}%)')
```

The with `torch.no_grad():` context tells PyTorch not to calculate gradient of tensor operations within it. Since we don't need to calculate gradient during testing this increases performance and decreases memory usage.


After getting the predictions from the model we get index of the highest score. The index indicates the digit. We compare the indexes with the ground truth with `torch.eq()` which returns `True` for a match and `False` otherwise. Finally we sum over the resulting tensor. All `True's` count as one and `False's` as zero. So the sum will be the number of correct predictions.

If you trained for 10 epochs then your accuracy should be about 98%. Which is pretty good.

# Full Code

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

trainset = datasets.MNIST('dataset/', train=True, download=True, transform=preprocess)
testset = datasets.MNIST('dataset/', train=False, download=True, transform=preprocess)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

nf = 32 # conv layer sizes or number of filters/kernels
lr = 0.0001
beta1 = 0.5
beta2 = 0.999
device = "cuda" if torch.cuda.is_available() else "cpu"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(1, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # result will be 14x14

            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # result will be 7x7
        )

        self.linears = nn.Sequential(
            nn.Linear(1568, 100), # width=7, height=7, depth/filters/kernels=32 ; linear_input=7*7*32=1568
            nn.ReLU(),

            nn.Linear(100, 50),
            nn.ReLU(),

            nn.Linear(50, 10),
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1) # flattening, result will be (64, 1568)
        x = self.linears(x)
        return x

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
criterion_CE = nn.CrossEntropyLoss().to(device)

model.train()
epoch = 10
for e in range(epoch):
    print(f'Starting epoch {e} of {epoch}')
    for X, y in trainloader:
        X = X.to(device)
        predictions = model(X)
        optimizer.zero_grad()
        loss = criterion_CE(predictions, y)
        loss.backward()
        optimizer.step()
    print(f'Loss: {loss.item()}')

torch.save(model.state_dict(), "model.pt")

model.eval()
correct = 0
for X, y in testloader:
    with torch.no_grad():
        X = X.to(device)
        output = model(X)
        predictions = output.max(1)[1]
        correct += torch.eq(predictions, y).sum()


print(f'accuracy: {int(correct)}/{len(testloader.dataset)} ({int(correct)/len(testloader.dataset)} or {int(correct)/len(testloader.dataset) * 100}%)')
```

I hope you found this helpful. To learn more about PyTorch I highly recommend the official [documentation](https://pytorch.org/docs/stable/index.html).

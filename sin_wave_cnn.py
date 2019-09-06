# Author: Cyrus
# Aug 2019
#
# here is a useful link for ref: https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/


from __future__ import print_function

from torch.autograd import Variable
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import time


from sklearn.metrics import roc_auc_score
from torch.utils.data.sampler import SubsetRandomSampler

# Temporal Convolution Model
class TempConv(nn.Module):
    def __init__(self,batch_size):
        super(TempConv, self).__init__()

        # in shape: [batch, 1, length]
        # out shape: [batch, 5, length - kernel_size + 1]
        self.conv1 = T.nn.Conv1d(in_channels=1, out_channels=5, kernel_size=100, stride=1, padding=0)

        # out shape: [batch, 5, length - kernel_size + 1]
        self.pool = T.nn.MaxPool1d(kernel_size=100, stride=1, padding=0)

        # out shape: [batch, 5, length - kernel_size + 1]
        self.conv2 = T.nn.Conv1d(in_channels=5, out_channels=5, kernel_size=50, stride=1, padding=0)

        # out shape: [batch, 5, length - kernel_size + 1]
        self.pool2 = T.nn.MaxPool1d(kernel_size=50, stride=1, padding=0)

        # if input len is 100, here it is: 64
        self.fc1 = T.nn.Linear(in_features=5*704, out_features=32)

        self.fc2 = T.nn.Linear(in_features=32, out_features=2)

        self.batchSize = batch_size

    def forward(self,x):
        c1 = self.conv1(x)
        c1 = F.relu(c1)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        c2 = F.relu(c2)
        p2 = self.pool2(c2)

        f1 = self.fc1(p2.view(self.batchSize, 5 * 704))
        f1 = F.relu(f1)
        f2 = self.fc2(f1)

        return f2


def createLossAndOptimizer(net, learning_rate=0.001):
    # Loss function
    loss = T.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return (loss, optimizer)


def trainNet(net, batch_size, n_epochs, learning_rate,inputX,responseY):

    inputX = T.from_numpy(inputX)
    responseY = T.from_numpy(responseY)

    dataset = list(zip(inputX,responseY))

    # Training
    n_training_samples = 800
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

    # Test
    n_test_samples = 200
    test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))


    train_loader = T.utils.data.DataLoader(dataset[0:400] + dataset[500:900], batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = T.utils.data.DataLoader(dataset[400:500] + dataset[900:], batch_size=batch_size, sampler=test_sampler)

    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    # Get training data
    n_batches = len(train_loader)

    # Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)

    # Time for printing
    training_start_time = time.time()

    # Loop for n_epochs
    for epoch in range(n_epochs):

        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):

            # Get inputs
            inputs, labels = data

            inputs = inputs[:,None,:]
            labels = labels.long()

            # Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss_size.data[0]
            total_train_loss += loss_size.data[0]

            # Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        preds = []
        trues = []
        for inputs, labels in test_loader:

            inputs = inputs[:, None, :]
            labels = labels.long()
            # Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            # Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            aaa = val_outputs.detach().numpy()
            tmp = []
            for i in range(aaa.shape[0]):
                if aaa[i,0] < aaa[i,1]:
                    tmp.append(1)
                else:
                    tmp.append(0)
            preds = preds + tmp
            trues = trues + labels.detach().numpy().tolist()
            total_val_loss += val_loss_size.item()

        print(list(zip(trues,preds)))
        print("Validation loss = {:.2f}".format(total_val_loss / len(test_loader)))
        print(len(trues))
        print('AUROC ',roc_auc_score(trues,np.array(preds)))

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

if __name__ == '__main__':
    np.random.seed(0)
    T.manual_seed(0)
    dataX = T.load('time_series_x_100x.pt')
    dataY = T.load('time_series_y_100x.pt')

    model = TempConv(20)
    model.double()

    trainNet(model,20,1,0.001,dataX,dataY)

    # Visualize conv1 filters
    kernels = model.conv1.weight.detach()
    fig, axarr = plt.subplots(kernels.size(0))
    for idx in range(kernels.size(0)):
        print(kernels[idx].squeeze())
        axarr[idx].plot(list(range(100)),kernels[idx].squeeze().numpy())
    plt.show()

    # Visualize conv2 filters
    kernels = model.conv2.weight.detach()
    fig, axarr = plt.subplots(kernels.size(0))
    for idx in range(kernels.size(0)):

        axarr[idx].plot(list(range(50)), kernels[idx].numpy().transpose())
    plt.show()


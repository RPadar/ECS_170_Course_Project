'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluation_Metrics import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt


class Method_CNN_MNIST(method, nn.Module):
    data = None
    # defines the max rounds to train the model
    # baseline - 10
    max_epoch = 10
    # defines the learning rate for gradient descent based optimizer for model learning
    # baseline - 1e-3
    learning_rate = 1e-3
    # for mini-batch gradient descent
    # baseline - 64
    batch_size = 64

    # defines CNN model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # use mps if available
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # FIRST CONVOLUTIONAL BLOCK
        # baseline: 1 channel in, 32 channels out; 3x3 kernel w/ padding=1
        # apply convolution with 3x3 kernel/filter
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # add nonlinearity, cheap computation
        self.relu1 = nn.ReLU()
        # compress information
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # SECOND CONVOLUTIONAL BLOCK
        # baseline: 32 channels in, 64 channels out; 3x3 kernel w/ padding=1
        # apply convolution
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # nonlinear activation
        self.relu2 = nn.ReLU()
        # pooling/compression
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # CLASSIFICATION BLOCK - convert tensor of features into class probabilities
        # flatten tensor into 1d vector for FC layer
        self.flatten = nn.Flatten()
        # bottleneck 3136 features into 128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # nonlinear activation
        self.relu_fc = nn.ReLU()
        # drop activation to learn more robust model
        # avoids dependencies on specific neurons
        self.dropout = nn.Dropout(0.5)
        # final mapping to class probabilities
        self.fc2 = nn.Linear(128, 10)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x):
        '''Forward propagation'''
        # first convolutional block - conv1 -> relu1 -> pool1
        x = self.pool1(self.relu1(self.conv1(x)))
        # second convolutional block - conv2 -> relu2 -> pool2
        x = self.pool2(self.relu2(self.conv2(x)))

        # classification block - flatten -> fc1 w/ relu3 -> dropout regularization -> fc2
        x = self.flatten(x)
        x = self.relu_fc(self.fc1(x))
        # dropout regularization
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # history lists for plots
        self.loss_history = []
        self.accuracy_history = []

        # move model parameters to active device (mps if available)
        self.to(self.device)

        # enable training mode - enable dropout layer
        nn.Module.train(self, True)

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')


        # convert data to tensors and move to device
        X_tensor = torch.FloatTensor(np.array(X)).to(self.device)
        y_tensor = torch.LongTensor(np.array(y)).to(self.device)

        n_samples = len(X_tensor)

        # it will be an iterative gradient updating process
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...

            # random permutation of data for stochastic gradient descent
            permutation = torch.randperm(n_samples)

            # metrics per epoch
            epoch_loss = 0
            epoch_correct = 0

            # iterate through mini-batches
            for i in range(0, n_samples, self.batch_size):
                # get indices of current batch from permutation
                batch_indices = permutation[i:i + self.batch_size]
                X_batch = X_tensor[batch_indices]
                y_batch = y_tensor[batch_indices]

                # forward pass - make predictions and calculate loss via cross-entropy
                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)

                # backpropagation - update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # track loss across the whole epoch
                epoch_loss += loss.item() * len(batch_indices)
                epoch_correct += (y_pred.max(1)[1] == y_batch).sum().item()

            # get metrics averaged per sample for current epoch
            avg_loss = epoch_loss / n_samples
            avg_accuracy = epoch_correct / n_samples
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(avg_accuracy)

            if epoch % 1 == 0:
                print("Epoch:", epoch, "Accuracy:", avg_accuracy, "Loss", avg_loss)

    def test(self, X):
        # disable training mode - dropout does no operation
        nn.Module.train(self, False)
        # do the testing, and return the result
        X_tensor = torch.FloatTensor(np.array(X)).to(self.device)
        y_pred = self.forward(X_tensor)
        # convert output logits to predicted class
        return y_pred.max(1)[1].cpu()

    # creates convergence plot for Section 3.5
    def plot_convergence(self):
        # for x-axis, number epochs starting from 1
        epochs = range(1, len(self.loss_history) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, self.loss_history, 'b-')
        ax1.set_title('CNN MNIST Training Loss Convergence')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax2.plot(epochs, self.accuracy_history, 'g-')
        ax2.set_title('CNN MNIST Training Accuracy Convergence')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('../../result/stage_3_result/convergence_CNN_MNIST.png')
        plt.show()

    def run(self):
        print('method running...')
        print(f'    device: {self.device}')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        self.plot_convergence()

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}

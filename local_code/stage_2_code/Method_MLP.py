'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluation_Metrics import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt

class Method_MLP(method, nn.Module):
    data = None
    # defines the max rounds to train the model
    max_epoch = 30
    # defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    # for mini-batch gradient descent
    batch_size = 64

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 256)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(256, 10)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_2 = nn.Softmax(dim=1)

        # normalize data based on min and max of entire dataset
        self.data_min = None
        self.data_max = None

    # computes min and max of data
    def compute_normalizer(self, data):
        data_tensor = torch.FloatTensor(np.array(data))
        self.data_min = data_tensor.min().item()
        self.data_max = data_tensor.max().item()

    # normalizes data
    def normalize(self, data):
        data_tensor = torch.FloatTensor(np.array(data))
        return (data_tensor - self.data_min) / (self.data_max - self.data_min)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))
        # output layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_2(self.fc_layer_2(h))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        self.loss_history = []
        self.accuracy_history = []

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # normalize input data
        self.compute_normalizer(X)
        X_tensor = self.normalize(X)
        # normalize returns a torch.tensor already, but y must be converted separately
        y_tensor = torch.LongTensor(np.array(y))
        n_samples = len(X_tensor)


        # it will be an iterative gradient updating proces
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...

            # random permutation of data for stochastic gradient descent
            permutation = torch.randperm(n_samples)

            # metrics
            epoch_loss = 0
            epoch_correct = 0

            # iterate through mini-batches
            for i in range(0, n_samples, self.batch_size):
                batch_indices = permutation[i:i + self.batch_size]
                X_batch = X_tensor[batch_indices]
                y_batch = y_tensor[batch_indices]

                # forward pass - make predictions and calculate loss
                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)

                # backpropagation - update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # track loss across the whole epoch
                epoch_loss += loss.item() * len(batch_indices)
                epoch_correct += (y_pred.max(1)[1] == y_batch).sum().item()

            # get metrics averaged per sample
            avg_loss = epoch_loss / n_samples
            avg_accuracy = epoch_correct / n_samples
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(avg_accuracy)

            if epoch % 10 == 0:
                print("Epoch:", epoch, "Accuracy:", avg_accuracy, "Loss", avg_loss)


    def test(self, X):
        # do the testing, and return the result
        y_pred = self.forward(self.normalize(X))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    # creates convergence plot for Section 3.5
    def plot_convergence(self):
        # for x-axis, number epochs starting from 1
        epochs = range(1, len(self.loss_history) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, self.loss_history, 'b-')
        ax1.set_title('Training Loss Convergence')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax2.plot(epochs, self.accuracy_history, 'g-')
        ax2.set_title('Training Accuracy Convergence')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('../../result/stage_2_result/convergence_curves.png')
        plt.show()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        self.plot_convergence()

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            
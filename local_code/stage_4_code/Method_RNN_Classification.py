'''
Concrete MethodModule class for Stage 4 -- RNN Text Classification on IMDB sentiment dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_4_code.Evaluation_Metrics import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt


class Method_RNN_Classification(method, nn.Module):
    data = None

    # TRAINING SETTINGS
    # defines the max rounds to train the model
    # baseline - 6
    max_epoch = 6
    # defines the learning rate for gradient descent based optimizer for model learning
    # baseline - 1e-3
    learning_rate = 1e-3
    # for mini-batch gradient descent
    # baseline - 64
    batch_size = 64

    # ARCHITECTURE SETTINGS
    # length of each word's embedding vector
    embed_dim = 128
    # width of hidden state
    hidden_dim = 128
    # number of stacked recurrent layers
    num_layers = 2
    # process data forward vs forward and backward
    bidirectional = True
    # dropout rate for dropout regularization before classification
    dropout = 0.5
    # choose last or mean pooling of hidden states before passing to classification layer
    pooling = 'last'

    # it defines the RNN model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, vocabulary_size=None, pad_index=0,
                 num_classes=2, cell_type='RNN'):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.cell_type = cell_type
        self.vocabulary_size = vocabulary_size
        self.pad_index = pad_index

        # use mps if available
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


        # EMBEDDING LAYER
        #
        # turn each word's id into learnable representative vector
        self.embedding = nn.Embedding(
            # number of embeddings to learn
            num_embeddings=vocabulary_size,
            # size of each embedding
            embedding_dim=self.embed_dim,
            # vector for pad character set to all 0s and not updated
            padding_idx=self.pad_index
        )

        # RECURRENT LAYER
        # build dictionary that can be passed into any cell type
        rnn_args = dict(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            # only use dropout between stacked recurrent layers (no dropout with only 1 layer)
            dropout=(self.dropout if self.num_layers > 1 else 0.0)
        )
        # only RNN PyTorch cells have nonlinearity parameter (LSTM and GRU hardcoded)
        if self.cell_type == 'RNN':
            self.rnn = nn.RNN(nonlinearity='tanh', **rnn_args)
        elif self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(**rnn_args)
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(**rnn_args)
        else:
            raise ValueError(f"unknown cell_type: {self.cell_type}")

        # CLASSIFIER LAYER - single fully-connected layer with dropout regularization
        # size of fc layer doubled if using bidirectional model concatenating forward and backward hidden states
        head_in = self.hidden_dim * (2 if self.bidirectional else 1)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(head_in, num_classes)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x):
        '''Forward propagation'''

        # get embeddings for words
        embedded = self.embedding(x)

        # output shape different for LSTM because it uses cell state
        # output contains hidden state at every time step
        # h_n = final hidden states, c_n = final cell states
        if self.cell_type == 'LSTM':
            output, (h_n, c_n) = self.rnn(embedded)
        else:
            output, h_n = self.rnn(embedded)

        # process hidden states after recurrent layer to determine what to pass to classification layer
        if self.pooling == 'last':
            # last pooling
            if self.bidirectional:
                # concatenate forward and backward processing RNN final hidden states for bidirectional model
                # h_n contains final hidden states for both
                recurrent_out = torch.cat((h_n[-2], h_n[-1]), dim=1)
            else:
                # otherwise just pull RNN's last hidden state
                recurrent_out = h_n[-1]
        else:
            # mean pooling - average hidden states across all timesteps
            recurrent_out = output.mean(dim=1)

        # classification layer
        recurrent_out = self.dropout_layer(recurrent_out)
        forward_out = self.fc(recurrent_out)
        return forward_out


    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here


    # model training method
    def train(self, X, y):
        # history lists for plots
        self.loss_history = []
        self.accuracy_history = []

        # use mps if available
        self.to(self.device)

        # enable training mode - enables dropout and batch normalization uses current batches statistics for mean and std
        nn.Module.train(self, True)

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        # Add L2 regularization to prevent overfitting
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # convert x and y data to tensors and send to device
        # LongTensor over FloatTensor because values are word ids not normalized pixel values
        X_tensor = torch.LongTensor(np.array(X)).to(self.device)
        y_tensor = torch.LongTensor(np.array(y)).to(self.device)
        n_samples = len(X_tensor)

        # it will be an iterative gradient updating proces
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...

            # random permutation of data for stochastic gradient descent
            permutation = torch.randperm(n_samples)

            # metrics
            epoch_loss = 0
            epoch_correct = 0

            # iterate through mini-batches
            for i in range(0, n_samples, self.batch_size):
                # get current batch
                batch_indices = permutation[i:i + self.batch_size]
                X_batch = X_tensor[batch_indices]
                y_batch = y_tensor[batch_indices]

                # forward pass - make predictions and calculate loss
                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)

                # backpropagation - update weights
                optimizer.zero_grad()
                loss.backward()
                # clip gradients to a max norm of 5.0 to protect against exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()

                # track loss across the whole epoch
                epoch_loss += loss.item() * len(batch_indices)
                epoch_correct += (y_pred.max(1)[1] == y_batch).sum().item()

                # report some batch statistics for each epoch
                if i % (self.batch_size * 100) == 0:
                    batch_accuracy = (y_pred.max(1)[1] == y_batch).float().mean().item()
                    print(f"  epoch {epoch} | sample {i}/{n_samples} |"
                          f" batch_loss {loss.item():.4f} | batch_accuracy {batch_accuracy:.4f}")

            # get metrics averaged per sample
            avg_loss = epoch_loss / n_samples
            avg_accuracy = epoch_correct / n_samples
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(avg_accuracy)

            # report epoch metrics
            if 1:
                print("Epoch:", epoch, "Accuracy:", avg_accuracy, "Loss", avg_loss)


    # model evaluation on test data with batched testing
    def test(self, X):
        # evaluation mode - no dropout
        nn.Module.train(self, False)
        X_tensor = torch.LongTensor(np.array(X)).to(self.device)

        predictions = []
        # disable gradient calculation during testing to preserve memory
        with torch.no_grad():
            # batch test set instead of making one forward pass over all testing data
            # avoid RunTimeError: Invalid buffer size
            for i in range(0, len(X_tensor), self.batch_size):
                X_batch = X_tensor[i:i+self.batch_size]
                y_pred = self.forward(X_batch)
                # argmax over logits gives predicted class
                predictions.append(y_pred.max(1)[1].cpu())

        # concat each batch's predictions together
        return torch.cat(predictions)


    # creates convergence plot for Section 3.5
    def plot_convergence(self):
        # for x-axis, number epochs starting from 1
        epochs = range(1, len(self.loss_history) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, self.loss_history, 'b-')
        ax1.set_title(f'{self.cell_type} Classification Training Loss Convergence')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax2.plot(epochs, self.accuracy_history, 'g-')
        ax2.set_title(f'{self.cell_type} Classification Training Accuracy Convergence')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'../../result/stage_4_result/convergence_{self.cell_type}_Classification.png')
        plt.show()


    def run(self):
        print('method running...')
        print(f'    device: {self.device}, cell: {self.cell_type}, pooling: {self.pooling}')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        self.plot_convergence()

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}

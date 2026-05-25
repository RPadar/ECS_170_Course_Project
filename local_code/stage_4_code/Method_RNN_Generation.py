'''
Concrete MethodModule class for Stage 4 -- RNN Text Generation on jokes dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method

import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt


class Method_RNN_Generation(method, nn.Module):
    data = None

    # TRAINING SETTINGS
    # defines the max rounds to train the model
    # baseline - 20
    max_epoch = 20
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
    # dropout rate for dropout regularization before classification and between recurrent layers
    dropout = 0.5

    # GENERATION SETTINGS
    # maximum number of words to generate after seed phrase
    # generation is also ended by <eos> token
    generation_length = 25
    # window_size must match the window_size set in the dataset loader
    # so it is passed from the data object to the method object by the script

    # it defines the RNN model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, vocabulary_size=None, pad_index=0,
                 cell_type='RNN'):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.cell_type = cell_type
        self.vocabulary_size = vocabulary_size
        self.pad_index = pad_index

        # use mps if available
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


        # EMBEDDING LAYER
        # turn each word's id into learnable representative vector using PyTorch's Embedding layer
        # gradient updates will flow to this layer during training
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

        # OUTPUT LAYER
        # output 1 probability score per vocabulary word
        head_in = self.hidden_dim
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(head_in, self.vocabulary_size)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x):
        '''Forward propagation - window of token ids to prediction score over vocabulary'''

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
        recurrent_out = h_n[-1]
        # classification layer
        # dropout then fc layer
        recurrent_out = self.dropout_layer(recurrent_out)
        forward_out = self.fc(recurrent_out)
        return forward_out


    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here


    # model training method
    def train(self, X, y):
        # history lists for plots
        self.loss_history = []
        # self.accuracy_history = []

        # use mps if available
        self.to(self.device)

        # enable training mode - enables dropout and batch normalization uses current batches statistics for mean and std
        nn.Module.train(self, True)

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()

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
            #epoch_correct = 0

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
                #epoch_correct += (y_pred.max(1)[1] == y_batch).sum().item()

                # report some batch statistics for each epoch
                if i % (self.batch_size * 100) == 0:

                    print(f"  epoch {epoch} | sample {i}/{n_samples} |"
                          f" batch_loss {loss.item():.4f}")

            # get metrics averaged per sample
            avg_loss = epoch_loss / n_samples
            self.loss_history.append(avg_loss)

            # report epoch metrics
            if 1:
                print("Epoch:", epoch, "Loss", avg_loss)

    # generates text from seed phrase
    # converts seed words to ids then predicts next word
    # adds next word to context window to predict next next word and so on
    # model chooses when to stop generation with <eos> token (or when reaching maximum generation length)
    def generate(self, seed, length=None):
        if length is None:
            length = self.generation_length

        # evaluation mode - disable dropout
        nn.Module.train(self, False)
        # send to device
        self.to(self.device)

        # get indices for unknown and eos tokens
        unknown_index = self.word_to_index['<unk>']
        eos_index = self.word_to_index['<eos>']

        # convert words of seed phrase to tokens/ids
        current_ids = []
        for word in seed:
            word = word.lower()
            current_ids.append(self.word_to_index.get(word, unknown_index))

        # disable gradient calculations during generation
        with torch.no_grad():
            for i in range(length):
                # context to generate is current window
                window = current_ids[-self.window_size:]
                x = torch.LongTensor([window]).to(self.device)

                logits = self.forward(x).squeeze(0)
                # predicted word is word with highest computed probability score / logit
                next_id = logits.argmax().item()
                # predicted word gets used for next context window
                current_ids.append(next_id)

                # end generation at end of stream token
                if next_id == eos_index:
                    break

        # convert tokens back to words and join into string
        words = [self.index_to_word[i] for i in current_ids]
        return ' '.join(words)


    # creates convergence plot for Section 3.5
    def plot_convergence(self):
        # for x-axis, number epochs starting from 1
        epochs = range(1, len(self.loss_history) + 1)

        plt.figure(figsize=(8,5))
        plt.plot(epochs, self.loss_history, 'b-')
        plt.title(f'{self.cell_type} Generation Training Loss Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'../../result/stage_4_result/convergence_{self.cell_type}_Generation.png')
        plt.show()


    def run(self):
        print('method running...')
        print(f'    device: {self.device}, cell: {self.cell_type}')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])

        self.plot_convergence()

        print('--generating samples')
        seeds = [['what', 'did', 'the'],
                 ['what', 'do', 'you'],
                 ['why', 'did', 'the'],
                 ['how', 'many', 'of']
                ]

        for seed in seeds:
            text = self.generate(seed)
            print(f"    seed {seed} ->\n    {text}\n")

        return {'loss_history': self.loss_history}

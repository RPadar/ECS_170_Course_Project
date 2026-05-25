'''
Concrete IO class for Stage 4 RNN generation dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import numpy as np

from local_code.base_class.dataset import dataset

# imports for language processing
import re
import string
from collections import Counter

import pandas as pd

# define padding, unknown, eos characters
_PADDING = '<pad>'
_UNKNOWN = '<unk>'
_EOS = '<eos>'

# keeps some punctuation for sentence structure / readability
_PUNCTUATION_KEPT = {'.', ',', '?', '!', "'"}
# puts space around punctuation to treat as separate "words" / tokens
_PUNCTUATION_SPACER = re.compile(r"([.,?!])")
# single string of punctuation to remove
_PUNCTUATION_DROPPED = ''.join(char for char in string.punctuation if char not in _PUNCTUATION_KEPT)
# translation table mapping punctuation to deletion ''
_PUNCTUATION_DROP_TABLE = str.maketrans('', '', _PUNCTUATION_DROPPED)


class Dataset_Loader_RNN_Generation(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    # HYPERPARAMETERS
    # context window
    # the number of words used as input to predict the next word
    window_size = 5
    # set number of words from training to keep in vocabulary
    # limit size of word embedding table
    max_vocab_size = 5000
    # minimum frequency for a word to appear to be added to vocabulary
    min_freq = 1

    # initialized as dictionaries storing mapping from words to embedding table indices and vice versa during load()
    word_to_index = None
    index_to_word = None


    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    # use pandas to read csv and return array of joke strings
    def _read_csv(self):
        path = self.dataset_source_folder_path + self.dataset_source_file_name
        df = pd.read_csv(path)
        data_instances = df['Joke'].tolist()
        return data_instances


    # non-public method to clean raw text data and return list of relevant words
    # cleans a single joke during _build_data_stream()
    def _clean(self, text):
        # normalize case
        text = text.lower()
        # add spacing around punctuation
        text = _PUNCTUATION_SPACER.sub(r' \1 ', text)
        # remove specified punctuation
        text = text.translate(_PUNCTUATION_DROP_TABLE)
        # split words/tokens by whitespace
        filtered_word_list = text.split()
        return filtered_word_list

    # clean each joke and concatenate into a single token stream
    def _build_data_stream(self, data):
        data_stream = []
        for instance in data:
            words = self._clean(instance)
            data_stream.extend(words)
            # insert <eos> after each joke to signify joke end
            data_stream.append(_EOS)
        return data_stream

    # build model's vocabulary as mapping of most frequent words to integers
    def _build_vocabulary(self, data_stream):
        # count word frequencies
        counter = Counter(data_stream)

        vocabulary = []
        for word, frequency in counter.most_common():
            # EOS added to vocabulary manually
            if frequency >= self.min_freq and word != _EOS:
                vocabulary.append(word)
        # cap vocabulary size
        vocabulary = vocabulary[:self.max_vocab_size]

        # insert special tokens into vocabulary
        self.word_to_index = {_PADDING: 0, _UNKNOWN: 1, _EOS: 2}

        # add vocabulary words to dictionary mappings
        for word in vocabulary:
            self.word_to_index[word] = len(self.word_to_index)
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}

    # divide continuous input stream into fixed-length windows associated with the immediately succeeding word/token
    # the next word/token is used as the label to predict during training
    # model makes predictions based on the currently examined window of input
    def _build_windows(self, data_stream):
        unknown_index = self.word_to_index[_UNKNOWN]
        # convert data stream elements from words to tokens/indexes
        word_tokens = []
        for token in data_stream:
            word_tokens.append(self.word_to_index.get(token, unknown_index))

        X, y = [], []

        for i in range(len(word_tokens) - self.window_size):
            X.append(word_tokens[i : i + self.window_size])
            y.append(word_tokens[i + self.window_size])

        return np.array(X, dtype=np.int64), np.array(y, dtype=np.int64)


    def load(self):
        print('loading RNN generation data...')

        # read and clean training and testing data
        print('--reading data instances')
        data_instances = self._read_csv()

        # single set of training data
        print('--cleaning data and building data stream')
        data_stream = self._build_data_stream(data_instances)

        # build vocabulary
        print('--building vocabulary')
        self._build_vocabulary(data_stream)
        print(f'    vocab size = {len(self.word_to_index)}')

       # make windows
        print('--making data stream windows for training')
        X, y = self._build_windows(data_stream)

        # return dict of data containing training data
        # + size of vocabulary and index for padding character needed for size of embedding and output layers
        return {
            'train': {'X': X, 'y': y},
            'vocabulary_size': len(self.word_to_index),
            'pad_index': self.word_to_index[_PADDING]
        }

# test script
if __name__ == '__main__':
    data_obj = Dataset_Loader_RNN_Generation('jokes', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation/'
    data_obj.dataset_source_file_name = 'data'
    loaded_data = data_obj.load()

    print('X shape:', loaded_data['train']['X'].shape)
    print('y shape:', loaded_data['train']['y'].shape)
    print('vocabulary_size:', loaded_data['vocabulary_size'])
    print('pad_index:', loaded_data['pad_index'])

    # show first window + succeeding word
    first_window = [data_obj.index_to_word[i] for i in loaded_data['train']['X'][0]]
    first_target = data_obj.index_to_word[loaded_data['train']['y'][0]]
    print(first_window, '->', first_target)

    # show punctuation was kept
    print(data_obj.word_to_index.get('?'))

    print(data_obj.word_to_index.get('!'))

    print(data_obj.word_to_index.get('<eos>'))

    print(data_obj.word_to_index.get('<unk>'))

    print(data_obj.word_to_index.get('what'))

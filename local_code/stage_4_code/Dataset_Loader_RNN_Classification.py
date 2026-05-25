'''
Concrete IO class for Stage 4 RNN classification dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import numpy as np

from local_code.base_class.dataset import dataset

# imports for language processing
import os
import re
import string
from collections import Counter

# natural language toolkit with pre-made stop words list
import nltk
from nltk.corpus import stopwords

# download stop words list if not installed
try:
    _NLTK_STOP_WORDS = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    _NLTK_STOP_WORDS = stopwords.words('english')

# define padding and unknown characters
_PADDING = '<pad>'
_UNKNOWN = '<unk>'

# define negation words to keep from stop words
# negation is important for sentiment classification
_NEGATIONS = {
    'no', 'not', 'nor', 'never', 'none', 'cannot',
    'don', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
    'haven', 'isn', 'mightn', 'mustn', 'needn', 'shouldn',
    'wasn', 'weren', 'wouldn', 'ain'
}

# remove negations from stop words list
_STOP_WORDS = set(_NLTK_STOP_WORDS) - _NEGATIONS

# create regex object to remove <br> html tags
_BR_TAG = re.compile(r'<br\s*/?>')

# define translation table of punctuation to remove
_PUNCTUATION = str.maketrans('', '', string.punctuation)



class Dataset_Loader_RNN_Classification(dataset):
    data = None
    dataset_source_folder_path = None

    # HYPERPARAMETERS
    # add padding or remove tokens such that each review is 150 tokens
    max_length = 150
    # set number of words from training to keep in vocabulary
    # limit size of word embedding table
    max_vocabulary_size = 25000
    # minimum frequency for a word to appear to be added to vocabulary
    min_freq = 2

    # will be initialized as dictionaries storing mapping from words to embedding table indices and vice versa
    word_to_index = None
    index_to_word = None


    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)


    # non-public method to clean raw text data and return list of relevant words
    def _clean(self, text):
        # remove <br> tags
        text = _BR_TAG.sub(' ', text)
        # normalize case
        text = text.lower()
        # remove punctuation
        text = text.translate(_PUNCTUATION)
        # split text by whitespace to get list of words
        text = text.split()

        # filter out non-alphabetic characters/words and stop words
        filtered_word_list = []
        for word in text:
            if word.isalpha() and word not in _STOP_WORDS:
                filtered_word_list.append(word)

        return filtered_word_list

    # reads and cleans each review for specified split (train or test)
    # pairs documents with corresponding training label based on their folder (neg or pos)
    def _read_split(self, split):
        documents, labels = [], []
        # pairing positive sentiment with 1 and negative with 0
        for label_name, label_value in (('pos', 1), ('neg', 0)):
            # go to pos or neg folder
            folder = os.path.join(self.dataset_source_folder_path, split, label_name)
            for file_name in os.listdir(folder):
                # skip over non-txt hidden files
                if not file_name.endswith('.txt'):
                    continue
                # gets path for file in folder
                path = os.path.join(folder, file_name)
                # open the review, clean it, append it to documents list
                with open(path, 'r', encoding='utf-8') as file:
                    documents.append(self._clean(file.read()))
                # append corresponding label to labels list
                labels.append(label_value)
        # return list of documents with matching list of labels
        return documents, labels

    # builds model's vocabulary - mapping of most frequent words from reviews to integers
    def _build_vocabulary(self, training_docs):
        # count frequency of each word in training documents
        counter = Counter()
        for document in training_docs:
            counter.update(document)

        # build vocabulary - keep words appearing more than min frequency and cap vocabulary at max size
        vocabulary = []
        # add most frequent words first
        for word, frequency in counter.most_common():
            if frequency >= self.min_freq:
                vocabulary.append(word)
        vocabulary = vocabulary[:self.max_vocabulary_size]

        # build dictionaries mapping words to indices and vice versa
        # At padding and unknown characters to model's vocabulary
        self.word_to_index = {_PADDING: 0, _UNKNOWN: 1}

        # add words from vocabulary to dictionary mappings in descending order of frequency at indices 2, 3, 4, ...
        for word in vocabulary:
            self.word_to_index[word] = len(self.word_to_index)
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}

    # converts review to numerical encoding and pads/truncates to specified size
    def _encode(self, docs):
        # get indices of padding and unknown characters
        pad_index = self.word_to_index[_PADDING]
        unknown_index = self.word_to_index[_UNKNOWN]

        # creates 2d array that will store encoded documents
        # encoding starts as all padding
        # actual words will overwrite padding characters and leftover padding will be necessary to reach max_length
        encoded_docs = np.full((len(docs), self.max_length), pad_index, dtype=np.int64)
        # iterate over each document
        for index, words in enumerate(docs):
            # truncate document to specified size
            words = words[: self.max_length]
            # iterate over each word within document
            for j, word in enumerate(words):
                # copy word's encoded value into encoded document
                # use unknown_index as encoded value if word is not in vocabulary
                encoded_docs[index, j] = self.word_to_index.get(word, unknown_index)

        return encoded_docs

    def load(self):
        print('loading RNN  classification data...')

        # read and clean training and testing data
        print('--reading and cleaning train')
        train_docs, train_labels = self._read_split('train')
        print('--reading and cleaning test')
        test_docs, test_labels = self._read_split('test')

        # build vocabulary
        print('--building vocabulary')
        self._build_vocabulary(train_docs)
        print(f'    vocab size = {len(self.word_to_index)}')

        # encode documents
        print('--encoding')
        X_train = self._encode(train_docs)
        X_test = self._encode(test_docs)
        y_train = np.array(train_labels, dtype=np.int64)
        y_test = np.array(test_labels, dtype=np.int64)

        # return dict of data containing train/test splits
        # + size of vocabulary and index for padding character needed for
        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test},
            'vocabulary_size': len(self.word_to_index),
            'pad_index': self.word_to_index[_PADDING]
        }


# run file as main to test data loader
if __name__ == '__main__':
    # load data
    data_obj = Dataset_Loader_RNN_Classification('imdb', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification'
    loaded_data = data_obj.load()

    print('X_train shape:', loaded_data['train']['X'].shape)
    print('y_train shape:', loaded_data['train']['y'].shape)
    print('X_test shape:', loaded_data['test']['X'].shape)
    print('y_test shape:', loaded_data['test']['y'].shape)
    print('vocabulary_size:', loaded_data['vocabulary_size'])
    print('pad_index:', loaded_data['pad_index'])

    # show first loaded file converted back to words
    decoded = [data_obj.index_to_word[i] for i in loaded_data['train']['X'][0] if i != loaded_data['pad_index']]
    print('first cleaned review:', ' '.join(decoded[:30]))

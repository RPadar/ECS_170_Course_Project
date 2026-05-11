'''
Concrete IO class for Stage 3 ORL Dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import numpy as np
import pickle
from local_code.base_class.dataset import dataset


class Dataset_Loader_ORL(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading ORL data...')

        # load file data
        raw_data = pickle.load(open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb'))

        def split_and_preprocess_data(data):
            X, y = [], []
            for instance in data:
                # normalize image pixel values
                image_matrix = np.array(instance['image'], dtype=np.float32) / 255.0
                # image is greyscale with 3 identical color channels
                # extract just 1 color channel
                image_matrix = image_matrix[:,:,0]
                # add channel dimension (1)
                image_matrix = np.expand_dims(image_matrix, axis=0)
                # x is image matrix, y is label
                X.append(image_matrix)
                y.append(instance['label'])
            # return as numpy array
            return np.array(X), np.array(y)

        # normalize and split data based on provided train/test split
        X_train, y_train = split_and_preprocess_data(raw_data['train'])
        X_test, y_test = split_and_preprocess_data(raw_data['test'])

        # return dictionary of data
        return {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}

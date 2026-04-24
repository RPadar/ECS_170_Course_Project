'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None

    # refactor to accommodate pre-split data
    dataset_source_file_name_train = None
    dataset_source_file_name_test = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')

        # helper function to load multiple csv files
        def load_csv(filename):
            X = []
            y = []
            f = open(self.dataset_source_folder_path + filename, 'r')

            for line in f:
                line = line.strip('\n')
                # changed to be comma separated
                elements = [int(i) for i in line.split(',')]
                # label is first element, features follow
                X.append(elements[1:])
                y.append(elements[0])

            f.close()
            return X, y

        X_train, y_train = load_csv(self.dataset_source_file_name_train)
        X_test, y_test = load_csv(self.dataset_source_file_name_test)

        # return dictionary of data
        return {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
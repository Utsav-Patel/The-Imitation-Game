import numpy as np
from tensorflow import keras
from helpers.helper import pre_process_input


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, input_data, output_data, batch_size=512, shuffle=True):
        'Initialization'
        self.input = input_data
        self.output = output_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.input))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        # self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.input) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        final_x = list()
        for dct in X:
            final_x.append(pre_process_input(dct['input'], dct['current_pos'], project_no=2, architecture_type='dense'))

        return np.array(final_x), y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.indexes = np.arange(len(self.input))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        return list(map(self.input.__getitem__, indexes)), self.output[indexes]

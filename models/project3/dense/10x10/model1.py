# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:21:46 2021

@author: Gambit
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from constants2 import NUM_ROWS, NUM_COLS, CHECKPOINT_FILEPATH, DATA_PATH, X, Y, NEIGHBOR_WEIGHT, CURRENT_CELL_WEIGHT,\
    VALIDATION_TEST_PATH
from model_architectures import create_model_project1_dense_20x20
from helpers.helper import check


def prepare_dataset(path):

    print('Loading data from', path)
    open_file = open(path, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()

    print("Successfully loaded data from pickle file")

    input_list = list()
    output_list = list()
    current_position_list = list()

    for dct in loaded_list:
        input_list.append(dct['input'])
        output_list.append(dct['output'])
        current_position_list.append(dct['current_pos'])

    for ind in range(len(input_list)):
        input_list[ind][current_position_list[ind][0]][current_position_list[ind][1]] = CURRENT_CELL_WEIGHT
        for ind2 in range(len(X)):
            neighbor = (current_position_list[ind][0] + X[ind2], current_position_list[ind][1] + Y[ind2])
            if check(neighbor, NUM_ROWS, NUM_COLS):
                input_list[ind][neighbor[0]][neighbor[1]] *= NEIGHBOR_WEIGHT

    input_numpy = np.array(input_list)
    input_numpy = input_numpy.reshape(input_numpy.shape[0], -1)
    # input_numpy = np.hstack((np.zeros(input_numpy.shape[0]).reshape(-1, 1), input_numpy))

    output_numpy = np.array(output_list)
    output_numpy = output_numpy.reshape(output_numpy.shape[0])
    output_numpy = to_categorical(output_numpy)

    return input_numpy, output_numpy


X_train, y_train = prepare_dataset(DATA_PATH)
X_val, y_val = prepare_dataset(VALIDATION_TEST_PATH)

# X_train, X_test, y_train, y_test = train_test_split(input_numpy, output_numpy, test_size=0.05, random_state=81)
X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.50, random_state=81)

print("X train shape", X_train.shape)
print("y train shape", y_train.shape)
print("X validation shape", X_val.shape)
print("y validation shape", y_val.shape)
print("X test shape", X_test.shape)
print("y test shape", y_test.shape)

model = create_model_project1_dense_20x20()
model.summary()

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH,
    verbose=1,
    save_weights_only=True,
    monitor='val_accuracy',
    save_best_only=False,
    save_freq='epoch'
)

history = model.fit(X_train, y_train, epochs=30, batch_size=512, validation_data=(X_val, y_val),
                    callbacks=[model_checkpoint_callback])

print(history.history)
model.evaluate(X_test,  y_test, verbose=2)

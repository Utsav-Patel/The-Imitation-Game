import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from constants import CHECKPOINT_FILEPATH, DATA_PATH, VALIDATION_TEST_PATH
from model_architectures import create_model_project1_cnn_20x20
from DataGenerator import DataGenerator


def prepare_dataset(path):
    open_file = open(path, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()

    print("Successfully loaded data from pickle file", path)

    input_list = list()
    output_list = list()

    for dct in loaded_list:
        input_list.append({'input': dct['input'], 'current_pos': dct['current_pos']})
        output_list.append(dct['output'])

    # input_numpy = np.array(input_list)
    # print(input_numpy.shape)
    # # input_numpy = input_numpy.reshape(input_numpy.shape[0], -1)

    output_numpy = np.array(output_list)
    output_numpy = output_numpy.reshape(output_numpy.shape[0])
    output_numpy = to_categorical(output_numpy)

    return input_list, output_numpy


# print("Input shape", input_numpy.shape)
# print("Output shape", output_numpy.shape)
# print('Starting training')

X_train, y_train = prepare_dataset(DATA_PATH)
X_val, y_val = prepare_dataset(VALIDATION_TEST_PATH)

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.50, random_state=81)

# print("X train shape", X_train.shape)
# print("y train shape", y_train.shape)
# print("X validation shape", X_val.shape)
# print("y validation shape", y_val.shape)
# print("X test shape", X_test.shape)
# print("y test shape", y_test.shape)

training_generator = DataGenerator(X_train, y_train)
validation_generator = DataGenerator(X_val, y_val)
testing_generator = DataGenerator(X_test, y_test)

model = create_model_project1_cnn_20x20()
model.summary()

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH,
    verbose=1,
    save_weights_only=True,
    monitor='val_accuracy',
    save_best_only=False,
    save_freq='epoch'
)

history = model.fit(training_generator, epochs=5, validation_data=validation_generator, use_multiprocessing=True,
                    workers=75, callbacks=[model_checkpoint_callback])

print(history.history)
model.evaluate(testing_generator, verbose=2)

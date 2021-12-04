import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from constants import NUM_ROWS, CHECKPOINT_FILEPATH, DATA_PATH
from model_architectures import create_model_project1_agent_10x10

open_file = open(DATA_PATH, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print("Successfully loaded")

input_list = list()
output_list = list()
current_position_list = list()

for dct in loaded_list:
    input_list.append(dct['input'])
    output_list.append(dct['output'])
    current_position_list.append(dct['current_pos'])

input_numpy = np.array(input_list)
input_numpy = input_numpy.reshape(input_numpy.shape[0], -1)
input_numpy = np.hstack((np.zeros(input_numpy.shape[0]).reshape(-1, 1), input_numpy))
for ind in range(input_numpy.shape[0]):
    input_numpy[ind][0] = current_position_list[ind][0] * NUM_ROWS + current_position_list[ind][1]

output_numpy = np.array(output_list)
output_numpy = output_numpy.reshape(output_numpy.shape[0])
output_numpy = to_categorical(output_numpy)

print("Input shape", input_numpy.shape)
print("Output shape", output_numpy.shape)
print('Starting training')

X_train, X_test, y_train, y_test = train_test_split(input_numpy, output_numpy, test_size=0.05, random_state=81)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=81)

print("X train shape", X_train.shape)
print("y train shape", y_train.shape)
print("X validation shape", X_val.shape)
print("y validation shape", y_val.shape)
print("X test shape", X_test.shape)
print("y test shape", y_test.shape)

model = create_model_project1_agent_10x10()
model.summary()

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH,
    verbose=1,
    save_weights_only=True,
    monitor='val_accuracy',
    save_best_only=False,
    save_freq='epoch'
)

history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_val, y_val),
                    callbacks=[model_checkpoint_callback])

print(history.history)
model.evaluate(X_test,  y_test, verbose=2)

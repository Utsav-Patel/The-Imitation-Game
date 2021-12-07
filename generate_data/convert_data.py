"""
This is the file to generate data.
"""

# Necessary Imports
import os.path
from datetime import datetime
import numpy as np
import pickle
import multiprocessing

from constants import NUM_ROWS, NUM_COLS, DATA_PATH, CURRENT_CELL_WEIGHT, NEIGHBOR_WEIGHT, X, Y
from helpers.helper import check

# Just to check how much time the code took
print('Start running this file at', datetime.now().strftime("%m-%d-%Y %H-%M-%S"))


def parallel_process(data: dict):
    position = np.zeros((NUM_ROWS, NUM_COLS))
    position[data['current_pos'][0]][data['current_pos'][1]] = CURRENT_CELL_WEIGHT
    for ind2 in range(len(X)):
        neighbor = (data['current_pos'][0] + X[ind2], data['current_pos'][1] + Y[ind2])
        if check(neighbor, NUM_ROWS, NUM_COLS):
            position[neighbor[0]][neighbor[1]] = NEIGHBOR_WEIGHT
    return {
        'input': np.stack(((data['input'] % 100) - 1, np.floor(data['input'] / 100), position)),
        'output': data['output']
    }


if __name__ == "__main__":
    # Used multiprocessing to parallelize processes
    n_cores = int(multiprocessing.cpu_count())
    print('Number of cores', n_cores)
    p = multiprocessing.Pool(processes=n_cores)

    final_data = list()

    open_file = open(DATA_PATH, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()

    results = p.imap_unordered(parallel_process, loaded_list)

    open_file = open(os.path.join(os.path.dirname(DATA_PATH), '20x20_final.pkl'), "wb")
    pickle.dump(list(results), open_file)
    open_file.close()

# Ending execution for this file. Now only plots are remaining
print('Ending running this file at', datetime.now().strftime("%m-%d-%Y %H-%M-%S"))

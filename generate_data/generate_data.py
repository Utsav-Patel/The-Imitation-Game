"""
This is the file to generate data.
"""

# Necessary Imports
from datetime import datetime
import numpy as np
import pickle
import random
import multiprocessing

from src.Maze import Maze
from helpers.helper import generate_grid_with_probability_p, repeated_forward, compute_heuristics, manhattan_distance
from constants import NUM_ROWS, NUM_COLS, STARTING_POSITION_OF_AGENT, GOAL_POSITION_OF_AGENT, INF, DATA_PATH, \
    PROJECT_NO, ARCHITECTURE_TYPE, VALIDATION_TEST_PATH

# Just to check how much time the code took
print('Start running this file at', datetime.now().strftime("%m-%d-%Y %H-%M-%S"))

start_value_of_probability = 0.17
end_value_of_probability = 0.33

num_uniform_samples = 14
num_times_run_for_each_probability = 1000

# List of probability values
list_of_probability_values = np.linspace(start_value_of_probability, end_value_of_probability, num_uniform_samples)


def parallel_process_for_each_probability(p):
    # Initialize attributes for this problem and compute heuristics
    maze = Maze(NUM_COLS, NUM_ROWS)
    compute_heuristics(maze, GOAL_POSITION_OF_AGENT, manhattan_distance)
    data = list()

    maze_array = generate_grid_with_probability_p(p)

    maze.reset_except_h()
    # Call to repeated forward A*
    repeated_forward(maze, maze_array, data, STARTING_POSITION_OF_AGENT, GOAL_POSITION_OF_AGENT, project_no=PROJECT_NO,
                     architecture_type=ARCHITECTURE_TYPE)
    return data


if __name__ == "__main__":
    # Used multiprocessing to parallelize processes
    n_cores = int(multiprocessing.cpu_count())
    print('Number of cores', n_cores)
    p = multiprocessing.Pool(processes=n_cores)

    final_data = list()

    # Iterate through each probability
    for probability_of_having_block in list_of_probability_values:

        # Just printing so we know where we are at execution
        print('Running for ', probability_of_having_block)

        results = p.imap_unordered(parallel_process_for_each_probability, [probability_of_having_block] *
                                   num_times_run_for_each_probability)

        for result in results:
            for dct in result:
                final_data.append(dct)

    categorise_list = [list(), list(), list(), list()]

    for dct in final_data:
        categorise_list[dct['output']].append(dct)

    minimum_class_size = INF
    for i in range(len(categorise_list)):
        minimum_class_size = min(minimum_class_size, len(categorise_list[i]))
        print("length of ", i, "th list: ", len(categorise_list[i]))

    final_list = list()

    for i in range(len(categorise_list)):
        final_list = final_list + random.sample(categorise_list[i], minimum_class_size)

    open_file = open(VALIDATION_TEST_PATH, "wb")
    pickle.dump(final_list, open_file)
    open_file.close()

# Ending execution for this file. Now only plots are remaining
print('Ending running this file at', datetime.now().strftime("%m-%d-%Y %H-%M-%S"))

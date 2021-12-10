import random
import tensorflow as tf
import os
import numpy as np
import multiprocessing
from datetime import datetime
from numba import cuda

from constants import CHECKPOINT_FILEPATH, NUM_ROWS, NUM_COLS, STARTING_POSITION_OF_AGENT, GOAL_POSITION_OF_AGENT,\
    STATE_OF_THE_ART_MODEL_PROJECT1_DENSE_CHECKPOINT_PATH, STATE_OF_THE_ART_MODEL_PROJECT1_CNN_CHECKPOINT_PATH
from src.Maze import Maze
from helpers.helper import generate_grid_with_probability_p, check, explore_neighbors, compute_heuristics,\
    manhattan_distance, repeated_forward, ml_agent_dfs, compute_trajectory_length_from_path

# Just to check how much time the code took
print('Start running this file at', datetime.now().strftime("%m-%d-%Y %H-%M-%S"))

start_value_of_probability = 0.0
end_value_of_probability = 0.30

num_uniform_samples = 31
num_times_run_for_each_probability = 1

# List of probability values
list_of_probability_values = np.linspace(start_value_of_probability, end_value_of_probability, num_uniform_samples)


def parallel_process_for_each_probability(p):
    # Initialize attributes for this problem and compute heuristics
    mazes = [Maze(NUM_COLS, NUM_ROWS), Maze(NUM_COLS, NUM_ROWS), Maze(NUM_COLS, NUM_ROWS)]
    compute_heuristics(mazes[0], GOAL_POSITION_OF_AGENT, manhattan_distance)

    full_maze = generate_grid_with_probability_p(p)

    # Call to repeated forward A*
    final_paths = repeated_forward(mazes[0], full_maze, None, STARTING_POSITION_OF_AGENT, GOAL_POSITION_OF_AGENT)[0]
    trajectory_length2 = ml_agent_dfs(mazes[2], full_maze, STARTING_POSITION_OF_AGENT, GOAL_POSITION_OF_AGENT,
                                      STATE_OF_THE_ART_MODEL_PROJECT1_CNN_CHECKPOINT_PATH, project_no=1,
                                      architecture_type='cnn')
    # cuda.select_device(0)
    # cuda.close()
    trajectory_length1 = ml_agent_dfs(mazes[1], full_maze, STARTING_POSITION_OF_AGENT, GOAL_POSITION_OF_AGENT,
                                      STATE_OF_THE_ART_MODEL_PROJECT1_DENSE_CHECKPOINT_PATH, project_no=1,
                                      architecture_type='dense')

    # cuda.select_device(0)
    # cuda.close()
    return [compute_trajectory_length_from_path(final_paths), trajectory_length1, trajectory_length2]


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

        print('Probability:', probability_of_having_block)
        print(list(results))
        # for result in results:
        #     for dct in result:
        #         final_data.append(dct)


# Ending execution for this file. Now only plots are remaining
print('Ending running this file at', datetime.now().strftime("%m-%d-%Y %H-%M-%S"))

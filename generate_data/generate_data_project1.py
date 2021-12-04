"""
This is the file to generate data.
"""

# Necessary Imports
from datetime import datetime
import numpy as np
import pickle
import random

from src.Maze import Maze
from helpers.helper import generate_grid_with_probability_p, repeated_forward, compute_heuristics, manhattan_distance
from constants import NUM_ROWS, NUM_COLS, STARTING_POSITION_OF_AGENT, GOAL_POSITION_OF_AGENT, INF, DATA_PATH

# Just to check how much time the code took
print('Start running this file at', datetime.now().strftime("%m-%d-%Y %H-%M-%S"))

# Initialize attributes for this problem and compute heuristics
maze = Maze(NUM_COLS, NUM_ROWS)
compute_heuristics(maze, GOAL_POSITION_OF_AGENT, manhattan_distance)
data = list()

start_value_of_probability = 0.0
end_value_of_probability = 0.80

num_uniform_samples = 100
num_times_run_for_each_probability = 2000

# List of probability values
list_of_probability_values = np.linspace(start_value_of_probability, end_value_of_probability, num_uniform_samples)

# Iterate through each probability
for probability_of_having_block in list_of_probability_values:

    # Just printing so we know where we are at execution
    print('Running for ', probability_of_having_block)

    # Run the same code multiple times
    for run_num in range(num_times_run_for_each_probability):

        # Generate maze randomly with each cell is blocked with probability of `probability_of_having_block`
        maze_array = generate_grid_with_probability_p(probability_of_having_block)

        maze.reset_except_h()
        # Call to repeated forward A*
        final_paths, total_explored_nodes = repeated_forward(maze, maze_array, data, STARTING_POSITION_OF_AGENT,
                                                             GOAL_POSITION_OF_AGENT)[:2]


categorise_list = [list(), list(), list(), list(), list()]

for dct in data:
    categorise_list[dct['output']].append(dct)

minimum_class_size = INF
for i in range(len(categorise_list)):
    minimum_class_size = min(minimum_class_size, len(categorise_list[i]))
    print("length of ", i, "th list: ", len(categorise_list[i]))

final_list = list()

for i in range(len(categorise_list)):
    final_list = final_list + random.sample(categorise_list[i], minimum_class_size)

open_file = open(DATA_PATH, "wb")
pickle.dump(final_list, open_file)
open_file.close()

# Ending execution for this file. Now only plots are remaining
print('Ending running this file at', datetime.now().strftime("%m-%d-%Y %H-%M-%S"))
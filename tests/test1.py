import tensorflow as tf
import numpy as np
from datetime import datetime
import os
import pickle

from constants import NUM_ROWS, NUM_COLS, STARTING_POSITION_OF_AGENT, GOAL_POSITION_OF_AGENT, PROJECT_PATH, \
    STATE_OF_THE_ART_MODEL_PROJECT1_DENSE_CHECKPOINT_PATH, STATE_OF_THE_ART_MODEL_PROJECT1_CNN_CHECKPOINT_PATH
from src.Maze import Maze
from helpers.helper import generate_grid_with_probability_p, compute_heuristics, manhattan_distance, repeated_forward, \
    ml_agent_dfs, compute_trajectory_length_from_path, repeated_forward_astar
from model_architectures import create_model_project1_cnn_20x20, create_model_project1_dense_20x20

# Just to check how much time the code took
print('Start running this file at', datetime.now().strftime("%m-%d-%Y %H-%M-%S"))

start_value_of_probability = 0.10
end_value_of_probability = 0.30

num_uniform_samples = 21
num_times_run_for_each_probability = 10

# checkpoint_dir = os.path.dirname(CHECKPOINT_FILEPATH)
# latest = tf.train.latest_checkpoint(checkpoint_dir)

latest = tf.train.latest_checkpoint(STATE_OF_THE_ART_MODEL_PROJECT1_DENSE_CHECKPOINT_PATH)
model1 = create_model_project1_dense_20x20()
model1.load_weights(latest)

latest = tf.train.latest_checkpoint(STATE_OF_THE_ART_MODEL_PROJECT1_CNN_CHECKPOINT_PATH)
model2 = create_model_project1_cnn_20x20()
model2.load_weights(latest)

# List of probability values
list_of_probability_values = np.linspace(start_value_of_probability, end_value_of_probability, num_uniform_samples)


def parallel_process_for_each_probability(p):
    # Initialize attributes for this problem and compute heuristics
    mazes = [Maze(NUM_COLS, NUM_ROWS), Maze(NUM_COLS, NUM_ROWS), Maze(NUM_COLS, NUM_ROWS)]
    compute_heuristics(mazes[0], GOAL_POSITION_OF_AGENT, manhattan_distance)

    is_solvable = 0

    # Call to repeated forward A*
    print('Running A*')

    while is_solvable == 0:
        mazes[0].reset_except_h()
        full_maze = generate_grid_with_probability_p(p)
        final_paths, is_solvable = repeated_forward(mazes[0], full_maze, None, STARTING_POSITION_OF_AGENT,
                                                    GOAL_POSITION_OF_AGENT)[:2]
    print('Running for dense')
    data1 = ml_agent_dfs(mazes[1], full_maze, STARTING_POSITION_OF_AGENT, GOAL_POSITION_OF_AGENT, project_no=1,
                         architecture_type='dense', model=model1)
    print('Running for cnn')
    data2 = ml_agent_dfs(mazes[2], full_maze, STARTING_POSITION_OF_AGENT, GOAL_POSITION_OF_AGENT, project_no=1,
                         architecture_type='cnn', model=model2)
    print('Running for comparison')
    comparison = repeated_forward_astar(full_maze, STARTING_POSITION_OF_AGENT, GOAL_POSITION_OF_AGENT, model1, model2)

    return [compute_trajectory_length_from_path(final_paths), data1, data2, comparison]


if __name__ == "__main__":

    # Iterate through each probability
    final_list = list()
    for probability_of_having_block in list_of_probability_values:
        probability_list = list()
        for ind in range(num_times_run_for_each_probability):
            # Just printing so we know where we are at execution
            print('Probability:', probability_of_having_block, 'Number of run:', ind)
            result = parallel_process_for_each_probability(probability_of_having_block)
            probability_list.append(result)
            print(result)
        final_list.append(probability_list)

    with open(os.path.join(PROJECT_PATH, 'results', 'project2.pkl'), 'wb') as f:
        pickle.dump({'probability': list_of_probability_values, 'output': final_list}, f)
# Ending execution for this file. Now only plots are remaining
print('Ending running this file at', datetime.now().strftime("%m-%d-%Y %H-%M-%S"))

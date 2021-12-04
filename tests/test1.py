import tensorflow as tf
import os
import numpy as np

from constants import CHECKPOINT_FILEPATH, NUM_ROWS, NUM_COLS, STARTING_POSITION_OF_AGENT, X, Y, GOAL_POSITION_OF_AGENT
from src.Maze import Maze
from helpers.helper import generate_grid_with_probability_p, check
from model_architectures import create_model_project1_agent_10x10


checkpoint_dir = os.path.dirname(CHECKPOINT_FILEPATH)
latest = tf.train.latest_checkpoint(checkpoint_dir)
model = create_model_project1_agent_10x10()
model.load_weights(latest)

current_position = STARTING_POSITION_OF_AGENT
maze = Maze(NUM_ROWS, NUM_COLS)
full_maze = generate_grid_with_probability_p(0.25)


def create_input():
    array = maze.maze_numpy.copy().flatten()
    array = np.insert(array, 0, current_position[0] * NUM_ROWS + current_position[1]).reshape(1, -1)
    return array


while True:
    print(full_maze)
    action = model.predict(create_input())
    print(action)
    action = np.argmax(action)
    print(action)
    if action == 5:
        print('Maze is not solvable')
        break
    next_position = (current_position[0] + X[action-1], current_position[1] + Y[action-1])

    if check(next_position, NUM_COLS, NUM_ROWS):
        if full_maze[next_position[0]][next_position[1]] == 1:
            maze.maze_numpy[next_position[0]][next_position[1]] += 1
        else:
            current_position = next_position
            print('Current action', action)
            print('Next position', current_position)
    else:
        print("Out of maze")

    if current_position == GOAL_POSITION_OF_AGENT:
        print('Target reached')
        break
    input()

print('Completed')

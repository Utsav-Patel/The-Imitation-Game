import random
import tensorflow as tf
import os
import numpy as np

from constants import CHECKPOINT_FILEPATH, NUM_ROWS, NUM_COLS, STARTING_POSITION_OF_AGENT, X, Y, GOAL_POSITION_OF_AGENT,\
    NEIGHBOR_WEIGHT, CURRENT_CELL_WEIGHT, TARGET_CANNOT_BE_REACHED_NUMBER, BLOCKED_NUMBER
from src.Maze import Maze
from helpers.helper import generate_grid_with_probability_p, check, explore_neighbors
from model_architectures import create_model_project1_dense_10x10


checkpoint_dir = os.path.dirname(CHECKPOINT_FILEPATH)
latest = tf.train.latest_checkpoint(checkpoint_dir)
model = create_model_project1_dense_10x10()
model.load_weights(latest)

current_position = STARTING_POSITION_OF_AGENT
maze = Maze(NUM_ROWS, NUM_COLS)
full_maze = generate_grid_with_probability_p(0.30)


def create_input():
    array = maze.maze_numpy.copy()
    array[current_position[0]][current_position[1]] = CURRENT_CELL_WEIGHT

    for ind2 in range(len(X)):
        neighbour = (current_position[0] + X[ind2], current_position[1] + Y[ind2])
        if check(neighbour, NUM_ROWS, NUM_COLS):
            array[neighbour[0]][neighbour[1]] *= NEIGHBOR_WEIGHT
    # array = np.insert(array, 0, current_position[0] * NUM_ROWS + current_position[1]).reshape(1, -1)
    return array.reshape(1, -1)


while True:
    print(full_maze)
    explore_neighbors(maze, full_maze, current_position)
    print(maze.maze_numpy)
    # Exploration

    action = model.predict(create_input())
    print(action)
    action = random.choices(np.arange(len(action[0])), action[0])[0]
    print(action)
    if action == TARGET_CANNOT_BE_REACHED_NUMBER:
        print('Maze is not solvable')
        break
    next_position = (current_position[0] + X[action], current_position[1] + Y[action])

    if check(next_position, NUM_COLS, NUM_ROWS):
        if full_maze[next_position[0]][next_position[1]] == BLOCKED_NUMBER:
            maze.maze_numpy[next_position[0]][next_position[1]] += 1*BLOCKED_NUMBER
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

# Import necessary things
from abc import ABC, abstractmethod

import numpy as np

from constants import NUM_ROWS, NUM_COLS, STARTING_POSITION_OF_AGENT, X, Y, UNVISITED_NUMBER
from src.Cell import Cell
from helpers.helper import astar_search_list, check


# Agent class
class Agent(ABC):
    # Method to initialize some variables for this class
    def __init__(self):
        self.maze = list()
        # Compute four and eight neighbors of each cell
        for row in range(NUM_ROWS):
            cell_list = list()
            for col in range(NUM_COLS):
                cell_list.append(Cell())
                for ind in range(len(X)):
                    neighbor = (row + X[ind], col + Y[ind])
                    if check(neighbor, NUM_ROWS, NUM_COLS):
                        cell_list[col].four_neighbors.append(neighbor)
                for ind in range(-1, 2):
                    for ind2 in range(-1, 2):
                        neighbor = (row + ind, col + ind2)
                        if neighbor == (row, col):
                            continue
                        if check(neighbor, NUM_ROWS, NUM_COLS):
                            cell_list[col].eight_neighbors.append(neighbor)
            self.maze.append(cell_list)

        # Initialize some important variables of this class
        self.current_position = STARTING_POSITION_OF_AGENT
        self.maze_numpy = np.zeros((NUM_ROWS, NUM_COLS)) + UNVISITED_NUMBER
        self.final_paths = list()
        self.parents = dict()
        self.children = dict()
        self.current_estimated_goal = list()
        self.num_examinations = 0

        self.maze_exam_numpy = np.zeros((NUM_ROWS, NUM_COLS))

        self.num_confirmed_cells = 0
        self.num_confirmed_blocked_cells = 0

        self.num_astar_calls = 0
        self.num_bumps = 0
        self.num_early_termination = 0

        # set list of global threshold
        # self.global_threshold = list()
        # self.global_threshold.append(math.ceil(math.log(ACCURACY_TO_ACHIEVE) / math.log(FLAT_FALSE_NEGATIVE_RATE)))
        # self.global_threshold.append(math.ceil(math.log(ACCURACY_TO_ACHIEVE) / math.log(HILLY_FALSE_NEGATIVE_RATE)))
        # self.global_threshold.append(math.ceil(math.log(ACCURACY_TO_ACHIEVE) / math.log(FOREST_FALSE_NEGATIVE_RATE)))

    # General method for planning
    def planning(self, goal_pos):
        """
        Method to find a path from current position to current estimated goal
        :param goal_pos: agent's current estimated goal
        :return: Nothing as we are storing results in agent's parents object
        """
        self.parents, num_explored_nodes = astar_search_list(self.maze, self.current_position, goal_pos)[:2]
        # self.num_cells_processed_while_planning += num_explored_nodes

    # reset method
    def reset(self):
        """
        Reset method to reset each each variable
        :return: Nothing
        """
        self.current_position = STARTING_POSITION_OF_AGENT
        self.final_paths = list()
        self.parents = dict()
        self.children = dict()
        self.current_estimated_goal = list()

        self.maze_numpy = np.zeros((NUM_ROWS, NUM_COLS)) + UNVISITED_NUMBER
        self.maze_exam_numpy = np.zeros((NUM_ROWS, NUM_COLS))

        self.num_confirmed_cells = 0
        self.num_confirmed_blocked_cells = 0

        self.num_astar_calls = 0
        self.num_bumps = 0
        self.num_early_termination = 0

        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                self.maze[row][col].reset()

        self.num_examinations = 0

    @abstractmethod
    def execution(self, full_maze: np.array, target_pos: tuple = None):
        pass

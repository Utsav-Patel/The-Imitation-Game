# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 10:25:11 2021

@author: Gambit
"""

import time
import numpy as np
import multiprocessing
from datetime import datetime
import pickle

from constants import STARTING_POSITION_OF_AGENT, INF, PROBABILITY_OF_GRID, NUM_ROWS, NUM_COLS, NUM_ITERATIONS
from helpers.helper import generate_grid_with_probability_p, compute_explored_cells_from_path, \
    length_of_path_from_source_to_goal, examine_and_propagate_probability, generate_target_position
from src.Agent6 import Agent6


agent = Agent6()


def find_the_target(num: int):
    """
    Function to run each process for each grid
    :param num: number of times it's running
    :return: [total movements, total examinations, total actions]
    """
    print('Running for:', num)

    agents = [6]
    x = list()
    data = list()
    # Keep generating grid and target position until we will get valid pair of it
    while True:
        random_maze = generate_grid_with_probability_p(PROBABILITY_OF_GRID)
        target_pos = generate_target_position(random_maze)
        if length_of_path_from_source_to_goal(random_maze, STARTING_POSITION_OF_AGENT, target_pos) != INF:
            break

    # Run agent 6,7, and 8 for the above generate grid and target position
    for agent_num in agents:

        # Print when the agent started it's execution
        print('Starting agent', agent_num)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("date and time =", dt_string)

        # Reset agent before using it
        agent.reset()
        target_found = False

        # Run the following loop until target is not found
        while not target_found:

            # First, find the current estimated target
            agent.pre_planning(agent_num)

            # Second, prepare a path to reach to this target
            agent.planning(agent.current_estimated_goal)

            # If the given target is not reachable, set it's probability of containing target to zero and find another
            # target and it's corresponding path
            while agent.current_estimated_goal not in agent.parents:
                agent.maze[agent.current_estimated_goal[0]][agent.current_estimated_goal[1]].is_blocked = True
                examine_and_propagate_probability(agent.maze, agent.probability_of_containing_target,
                                                  agent.false_negative_rates, agent.current_position, target_pos,
                                                  agent.current_estimated_goal, agent.current_estimated_goal)
                agent.pre_planning(agent_num)
                agent.planning(agent.current_estimated_goal)

            # Execute on the generated path
            agent.execution(random_maze)

            # Examine the current cell
            target_found = agent.examine(target_pos)

        # Find total number of movements
        movements = compute_explored_cells_from_path(agent.final_paths)
        x.append([agent.num_examinations, movements])

        # End the execution of the current run
        print('ending agent', agent_num)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("date and time =", dt_string)
    return x


if __name__ == "__main__":

    start_time = time.time()

    # Used multiprocessing to parallelize processes
    n_cores = int(multiprocessing.cpu_count())
    print('Number of cores', n_cores)
    p = multiprocessing.Pool(processes=n_cores)

    results = p.imap_unordered(find_the_target, range(NUM_ITERATIONS))


    end_time = time.time()



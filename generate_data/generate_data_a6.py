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
import random

from constants2 import STARTING_POSITION_OF_AGENT, INF, PROBABILITY_OF_GRID, NUM_ROWS, NUM_COLS, NUM_ITERATIONS, X, Y, DATA_PATH, VALIDATION_TEST_PATH
from helpers.Agent6helper import generate_grid_with_probability_p, compute_explored_cells_from_path, \
    length_of_path_from_source_to_goal, examine_and_propagate_probability, generate_target_position
from src.Agent6 import Agent6

        # Print when the agent started it's execution
print('Starting agent 8')
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)
agent = Agent6()

start_value_of_probability = 0.21
end_value_of_probability = 0.30

num_uniform_samples = 10
num_times_run_for_each_probability = 1000

# List of probability values
list_of_probability_values = np.linspace(start_value_of_probability, end_value_of_probability, num_uniform_samples)

def find_the_target(p):
    """
    Function to run each process for each grid
    :param num: number of times it's running
    :return: [total movements, total examinations, total actions]
    """
    #print('Running for:', num)

    agents = [8]
    data = list()
    # Keep generating grid and target position until we will get valid pair of it
    while True:
        random_maze = generate_grid_with_probability_p(p)
        target_pos = generate_target_position(random_maze)
        if length_of_path_from_source_to_goal(random_maze, STARTING_POSITION_OF_AGENT, target_pos) != INF:
            break
    #print(target_pos)
    # Run agent 6,7, and 8 for the above generate grid and target position
    for agent_num in agents:



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
            agent.execution(random_maze, data)

            # Examine the current cell
            target_found = agent.examine(target_pos, data)




    return data

#result = find_the_target(1)




if __name__ == "__main__":

    start_time = time.time()

    # Used multiprocessing to parallelize processes
    n_cores = int(multiprocessing.cpu_count())
    print('Number of cores', n_cores)
    p = multiprocessing.Pool(processes=n_cores)
    final_data = list()
    
    for probability_of_having_block in list_of_probability_values:

        # Just printing so we know where we are at execution
        print('Running for ', probability_of_having_block)
        results = p.imap_unordered(find_the_target, [probability_of_having_block] * num_times_run_for_each_probability)
        
        
        for result in results:
                for dct in result:
                    final_data.append(dct)

    categorise_list = [list(), list(), list(), list(), list()]
    
    
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
    
    # End the execution of the current run
    print('ending agent 8')
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)

    end_time = time.time()




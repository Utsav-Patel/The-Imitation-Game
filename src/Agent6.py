# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 18:10:28 2021

@author: Gambit
"""

# Necessary imports
import numpy as np

from helpers.Agent6helper import forward_execution
from src.Agent import Agent
from helpers.Agent6helper import examine_and_propagate_probability, parent_to_child_dict, length_of_path_from_source_to_all_nodes
from constants2 import ONE_PROBABILITY, INF

# Blindfolded agent's class
class Agent6(Agent):
    def __init__(self):
        super().__init__()

    # Override execution method of Agent class
    def execution(self, full_maze: np.array, data, target_pos: tuple = None):
        """
        Agent 6,7, and 8's execution method
        :param full_maze: origin maze array
        :param target_pos: actual target position
        :return: Nothing
        """

        # Calculate children from parents dictionary
        self.children = parent_to_child_dict(self.parents, self.current_estimated_goal)
        #print(self.current_position, '        ', self.children)
        # Agent will move along the planned path to reach current estimated goal
        current_path = forward_execution(self.maze, self.false_negative_rates,self.maze_numpy, full_maze,
                                                         self.current_position, self.current_estimated_goal,
                                                         self.children, data, self.probability_of_containing_target,
                                                         self.maze_exam_numpy)
        # Pick the last element of the current path to get agent's current position
        #print('Current path')
        #print(current_path)
        #print(current_path[-1])
        self.current_position = current_path[-1]

        # Append current path to final path
        self.final_paths.append(current_path)

    def examine(self, target_pos, data, project_no = 3, architecture_type = 'dense'):
        """
        Method is used to examine the agent's current cell if it has reached to current estimated goal otherwise set
        next cell to block because we can't be able to reach because of that.
        :param target_pos: target's position
        :return: True if agent has found out target otherwise False
        """
        
        if self.current_position == self.current_estimated_goal:
            self.num_examinations += 1  
            self.maze_exam_numpy[self.current_position[0]][self.current_position[1]] += 1
            
            probability_of_finding_target = np.multiply(self.probability_of_containing_target,
                                                ONE_PROBABILITY - self.false_negative_rates)
            distance_array = length_of_path_from_source_to_all_nodes(self.maze, self.current_position)
            distance_array[self.current_position[0]][self.current_position[1]] = INF
            utility_function = np.divide(probability_of_finding_target, distance_array)
            if data is not None:
                if (project_no == 3) and (architecture_type == 'dense'):
                    data.append({
                        'current_pos': self.current_position,
                        'input': np.stack((self.maze_numpy.copy(), utility_function.copy()*1000)),
                        'output': 4
                    })
        is_target_found = examine_and_propagate_probability(self.maze, self.probability_of_containing_target,
                                                            self.false_negative_rates, self.current_position,
                                                            target_pos, self.current_estimated_goal,
                                                            self.children[self.current_position])

        return is_target_found

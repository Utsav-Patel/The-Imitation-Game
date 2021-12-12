# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 18:18:04 2021

@author: Gambit
"""

# Necessary imports

from constants import INF


# Cell class
class Cell:
    # Initialise variables for the cell class
    def __init__(self):
        self.g = INF
        self.h = INF
        self.f = INF

        self.is_blocked = False
        self.is_visited = False
        self.is_confirmed = False

        self.num_neighbor = 0
        self.min_hidden_cell_neighbor = INF

        self.num_confirmed_blocked = 0
        self.num_confirmed_unblocked = 0
        self.num_sensed_blocked = 0
        self.num_sensed_unblocked = 0

        self.probability_of_being_blocked = 0.0
        self.eight_neighbors = list()
        self.four_neighbors = list()
        self.eight_neighbors = list()

        # newly added properties
        #self.previous_examinations = 0
        #self.previous_visits = 0
        #self.max_threshold_of_examinations = 0

    # Reset attributes of this class
    def reset_except_h(self, default_probability: float = 0.0):
        self.g = INF
        self.f = INF

        self.is_blocked = False
        self.is_visited = False
        self.is_confirmed = False

        # self.num_neighbor = 0
        self.min_hidden_cell_neighbor = INF

        self.num_confirmed_blocked = 0
        self.num_confirmed_unblocked = 0
        self.num_sensed_blocked = 0
        self.num_sensed_unblocked = 0

        self.probability_of_being_blocked = default_probability

        #self.previous_examinations = 0
        #self.previous_visits = 0
        #self.max_threshold_of_examinations = 0

    def reset(self):
        self.reset_except_h()
        self.h = INF

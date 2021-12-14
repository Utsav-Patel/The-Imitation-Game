# Necessary imports
import numpy as np

from constants import GOAL_POSITION_OF_AGENT, NUM_ROWS, NUM_COLS, UNVISITED_NUMBER, UNBLOCKED_NUMBER, UNBLOCKED_WEIGHT,\
    BLOCKED_NUMBER, STARTING_POSITION_OF_AGENT
from src.Agent_Project2 import Agent
from helpers.helper import parent_to_child_dict, sense_current_node, find_block_while_inference, find_output


# Example inference class
class TheExampleInferenceAgent(Agent):
    def __init__(self):
        self.num_confirmed_blocked = np.zeros((NUM_ROWS, NUM_COLS))
        self.num_confirmed_unblocked = np.zeros((NUM_ROWS, NUM_COLS))
        self.num_sensed_blocked = np.zeros((NUM_ROWS, NUM_COLS))
        self.num_sensed_unblocked = np.zeros((NUM_ROWS, NUM_COLS))

        self.maze_numpy = np.zeros((NUM_ROWS, NUM_COLS)) + UNVISITED_NUMBER
        self.num_times_cell_visited = np.zeros((NUM_ROWS, NUM_COLS))

        self.current_estimated_goal = GOAL_POSITION_OF_AGENT

        super().__init__()

        self.maze_numpy[STARTING_POSITION_OF_AGENT[0]][STARTING_POSITION_OF_AGENT[1]] = UNBLOCKED_NUMBER
        self.num_times_cell_visited[STARTING_POSITION_OF_AGENT[0]][STARTING_POSITION_OF_AGENT[1]] += 1

    # Overridden execution method
    def execution(self, full_maze: np.array, target_position: tuple = None):

        data = list()
        # Increase counter of A* calls
        self.num_astar_calls += 1
        children = parent_to_child_dict(self.parents, GOAL_POSITION_OF_AGENT)
        current_position = self.current_position

        # Add all trajectory cells to one set
        entire_trajectory_nodes = set()
        entire_trajectory_nodes.add(current_position)

        while current_position != children[current_position]:
            current_position = children[current_position]
            entire_trajectory_nodes.add(current_position)

        current_position = self.current_position
        current_path = list()
        current_path.append(current_position)

        # Run this loop until we will reach at goal state or block cell
        while True:

            if current_position != self.current_position:
                self.maze_numpy[current_position[0]][current_position[1]] = UNBLOCKED_NUMBER
                self.num_times_cell_visited[current_position[0]][current_position[1]] += 1

            # Mark visited to unvisited cell and sense it
            if not self.maze[current_position[0]][current_position[1]].is_visited:
                self.maze[current_position[0]][current_position[1]].is_visited = True
                sense_current_node(self.maze, current_position, full_maze,
                                   num_confirmed_blocked=self.num_confirmed_blocked,
                                   num_sensed_blocked=self.num_sensed_blocked,
                                   num_confirmed_unblocked=self.num_confirmed_unblocked,
                                   num_sensed_unblocked=self.num_sensed_unblocked)

            # Check whether you can infer anything from the current node
            if find_block_while_inference(self.maze, current_position, full_maze, self.maze_numpy,
                                          self.num_confirmed_blocked, self.num_confirmed_unblocked,
                                          entire_trajectory_nodes):
                self.num_early_termination += 1
                break

            # Check whether the next node is block or not.
            if full_maze[children[current_position][0]][children[current_position][1]] == BLOCKED_NUMBER:
                find_block_while_inference(self.maze, children[current_position], full_maze, self.maze_numpy,
                                           self.num_confirmed_blocked, self.num_confirmed_unblocked)
                self.num_bumps += 1
                break
            else:
                if current_position == children[current_position]:
                    break

                if np.logical_and(self.num_confirmed_blocked >= 0, self.num_confirmed_blocked < 9).all() and \
                        np.logical_and(self.num_confirmed_unblocked >= 0, self.num_confirmed_unblocked < 9).all() and \
                        np.logical_and(self.num_sensed_blocked >= 0, self.num_sensed_blocked < 9).all() and \
                        np.logical_and(self.num_sensed_unblocked >= 0, self.num_sensed_unblocked < 9).all():

                    num_times_cell_visited = self.num_times_cell_visited.copy()
                    maze_numpy = self.maze_numpy.copy()
                    maze_numpy[maze_numpy == UNBLOCKED_NUMBER] *= UNBLOCKED_WEIGHT
                    data.append({
                        'current_pos': current_position,
                        'input': maze_numpy - num_times_cell_visited,
                        'sensed': 1 * self.num_confirmed_blocked + 10 * self.num_sensed_blocked +
                                  100 * self.num_confirmed_unblocked + 1000 * self.num_sensed_unblocked,
                        'output': find_output(current_position, children[current_position])
                    })
                else:
                    raise Exception('Values are not in particular range')
                # Add data here

                current_position = children[current_position]
                current_path.append(current_position)

        self.final_paths.append(current_path)
        self.current_position = current_path[-1]

        return data

    def reset(self):
        self.num_confirmed_blocked.fill(0)
        self.num_confirmed_unblocked.fill(0)
        self.num_sensed_blocked.fill(0)
        self.num_sensed_unblocked.fill(0)
        self.maze_numpy.fill(UNVISITED_NUMBER)
        self.num_times_cell_visited.fill(0)

        super().reset()

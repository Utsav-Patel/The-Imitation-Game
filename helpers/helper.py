import random
import numpy as np
from sortedcontainers import SortedSet
from queue import Queue

from src.Maze import Maze
from constants import NUM_COLS, NUM_ROWS, STARTING_POSITION_OF_AGENT, GOAL_POSITION_OF_AGENT, X, Y, UNBLOCKED_NUMBER, \
    BLOCKED_NUMBER, TARGET_CANNOT_BE_REACHED_NUMBER, CURRENT_CELL_WEIGHT, NEIGHBOR_WEIGHT, TRAINED_MODEL_NUM_ROWS, \
    TRAINED_MODEL_NUM_COLS, UNBLOCKED_WEIGHT, TRAJECTORY_LENGTH_THRESHOLD


def manhattan_distance(pos1: tuple, pos2: tuple):
    """
    Compute Manhattan distance between two points
    :param pos1: Coordinate of first point
    :param pos2: Coordinate of second point
    :return: Manhattan distance between two points
    """
    distance = 0
    for ind in range(len(pos1)):
        distance += abs(pos1[ind] - pos2[ind])
    return distance


def compute_heuristics(maze: Maze, goal_pos: tuple, h_func):
    """
    Compute Heuristic for the current maze
    :param maze: Maze
    :param goal_pos: This is the goal state where we want to reach
    :param h_func: Heuristic function we want to use
    :return: None as we are updating in the same maze object
    """

    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            if not maze.maze[row][col].is_blocked:
                maze.maze[row][col].h = h_func((row, col), goal_pos)


def compute_trajectory_length_from_path(paths: list):
    """
    This function will compute the trajectory length from the list of paths returned by any repeated forward algorithm
    :param paths: list of paths
    :return: trajectory length
    """

    trajectory_length = 0
    for path in paths:
        trajectory_length += len(path)
    trajectory_length -= len(paths)
    return trajectory_length


def generate_grid_with_probability_p(p):
    """
    This function will generate the uniform random grid of size NUM_ROWS X NUM_COLS.
    :param p: probability of cell being blocked
    :return: Grid of size NUM_ROWS X NUM_COLS with each cell having uniform probability of being blocked is p.
    """
    randomly_generated_array = np.random.uniform(low=0.0, high=1.0, size=NUM_ROWS * NUM_COLS).reshape(NUM_ROWS,
                                                                                                      NUM_COLS)
    randomly_generated_array[STARTING_POSITION_OF_AGENT[0]][STARTING_POSITION_OF_AGENT[1]] = UNBLOCKED_NUMBER
    randomly_generated_array[GOAL_POSITION_OF_AGENT[0]][GOAL_POSITION_OF_AGENT[1]] = UNBLOCKED_NUMBER
    randomly_generated_array[randomly_generated_array >= p] = UNBLOCKED_NUMBER
    randomly_generated_array[randomly_generated_array < p] = BLOCKED_NUMBER
    return randomly_generated_array


def check(pos: tuple, num_cols: int, num_rows: int):
    """
    Check whether current point is in the grid or not
    :param pos: current point
    :param num_cols: Number of columns of the grid
    :param num_rows: Number of rows of the grid
    :return: True if the current point is in the grid otherwise False
    """
    if (0 <= pos[0] < num_rows) and (0 <= pos[1] < num_cols):
        return True
    return False


def find_output(current_position: tuple, next_position: tuple):
    for ind in range(len(X)):
        if (current_position[0] + X[ind], current_position[1] + Y[ind]) == next_position:
            return ind

    raise Exception("Invalid Input")


def astar_search_list(maze: list, start_pos: tuple, goal_pos: tuple):
    """
    Function to compute A* search
    :param maze: maze is a list of list
    :param start_pos: starting position of the maze from where we want to start A* search
    :param goal_pos: goal position of the maze to where agent wants to reach
    :return: Returning the path from goal_pos to start_pos if it exists
    """

    # Initialize a set for visited nodes
    visited_nodes = set()

    # Initialize a sorted set to pop least value element from the set
    sorted_set = SortedSet()

    # Initialize a dictionary to store a random value assigned to each node. This dictionary would be helpful to know
    # the value of a node when we want to remove a particular node from the sorted set
    node_to_random_number_mapping = dict()

    # Initialize another dictionary to store parent information
    parents = dict()

    # Initialize g and f for the starting position
    maze[start_pos[0]][start_pos[1]].g = 0
    maze[start_pos[0]][start_pos[1]].h = manhattan_distance(start_pos, goal_pos)
    maze[start_pos[0]][start_pos[1]].f = maze[start_pos[0]][start_pos[1]].h

    # Assigning a random number to start position to the starting position and adding to visited nodes
    node_to_random_number_mapping[start_pos] = 0
    visited_nodes.add(start_pos)

    # Add start position node into the sorted set. We are giving priority to f(n), h(n), and g(n) in the decreasing
    # order. Push random number for random selection if there is conflict between two nodes
    # (If f(n), g(n), and h(n) are same for two nodes)
    sorted_set.add(((maze[start_pos[0]][start_pos[1]].f, maze[start_pos[0]][start_pos[1]].h,
                     maze[start_pos[0]][start_pos[1]].g, node_to_random_number_mapping[start_pos]), start_pos))

    parents[start_pos] = start_pos

    num_explored_nodes = 0

    # Running the loop until we reach our goal state or the sorted set is empty
    while sorted_set.__len__() != 0:
        # Popping first (shortest) element from the sorted set
        current_node = sorted_set.pop(index=0)

        # Increase the number of explored nodes
        num_explored_nodes += 1

        # If we have found the goal position, we can return parents and total explored nodes
        if current_node[1] == goal_pos:
            return parents, num_explored_nodes

        # Otherwise, we need to iterate through each child of the current node
        for val in range(len(X)):
            neighbour = (current_node[1][0] + X[val], current_node[1][1] + Y[val])

            # Neighbour should not go outside our maze and it should not be blocked if we want to visit that particular
            # neighbour
            if check(neighbour, NUM_ROWS, NUM_COLS) and (not maze[neighbour[0]][neighbour[1]].is_blocked):

                # If neighbour is being visited first time, we should change its g(n) and f(n) accordingly. Also, we
                # need to assign a random value to it for the time of conflict. In the end, we will add all those things
                # into the sorted set and update its parent
                if neighbour not in visited_nodes:
                    maze[neighbour[0]][neighbour[1]].g = maze[current_node[1][0]][current_node[1][1]].g + 1
                    maze[neighbour[0]][neighbour[1]].h = manhattan_distance(neighbour, goal_pos)
                    maze[neighbour[0]][neighbour[1]].f = maze[neighbour[0]][neighbour[1]].g + \
                                                         maze[neighbour[0]][neighbour[1]].h
                    node_to_random_number_mapping[neighbour] = val
                    visited_nodes.add(neighbour)
                    sorted_set.add(((maze[neighbour[0]][neighbour[1]].f,
                                     maze[neighbour[0]][neighbour[1]].h, maze[neighbour[0]][neighbour[1]].g,
                                     node_to_random_number_mapping[neighbour]), neighbour))
                    parents[neighbour] = current_node[1]

                # If a particular neighbour is already visited, we should compare its f(n) value to its previous f(n)
                # value. If current computed f(n) value is less than the previously computed value, we should remove
                # previously computed value and add new value to the sorted set
                else:
                    neighbour_g = maze[current_node[1][0]][current_node[1][1]].g + 1
                    neighbour_f = maze[neighbour[0]][neighbour[1]].h + neighbour_g
                    if neighbour_f < maze[neighbour[0]][neighbour[1]].f:

                        # The following if condition is needed only when the heuristic is inadmissible otherwise a
                        # neighbour has to be in the sorted set if we are able to find out less value of f(n) for that
                        # particular neighbour
                        if ((maze[neighbour[0]][neighbour[1]].f,
                             maze[neighbour[0]][neighbour[1]].h, maze[neighbour[0]][neighbour[1]].g,
                             node_to_random_number_mapping[neighbour]), neighbour) \
                                in sorted_set:
                            sorted_set.remove(
                                ((maze[neighbour[0]][neighbour[1]].f,
                                  maze[neighbour[0]][neighbour[1]].h, maze[neighbour[0]][neighbour[1]].g,
                                  node_to_random_number_mapping[neighbour]), neighbour))
                        maze[neighbour[0]][neighbour[1]].g = neighbour_g
                        maze[neighbour[0]][neighbour[1]].f = neighbour_f
                        node_to_random_number_mapping[neighbour] = val
                        sorted_set.add(
                            ((maze[neighbour[0]][neighbour[1]].f,
                              maze[neighbour[0]][neighbour[1]].h, maze[neighbour[0]][neighbour[1]].g,
                              node_to_random_number_mapping[neighbour]), neighbour))
                        parents[neighbour] = current_node[1]

    return parents, num_explored_nodes


def astar_search(maze: Maze, start_pos: tuple, goal_pos: tuple):
    """
    Function to compute A* search
    :param maze: Maze object
    :param start_pos: starting position of the maze from where we want to start A* search
    :param goal_pos: Goal state (position) where we want to reach
    :return: Returning the path from goal_pos to start_pos if it exists
    """

    # Initialize a set for visited nodes
    visited_nodes = set()

    # Initialize a sorted set to pop least value element from the set
    sorted_set = SortedSet()

    # Initialize a dictionary to store a random value assigned to each node. This dictionary would be helpful to know
    # the value of a node when we want to remove a particular node from the sorted set
    node_to_random_number_mapping = dict()

    # Initialize another dictionary to store parent information
    parents = dict()

    # Initialize g and f for the starting position
    maze.maze[start_pos[0]][start_pos[1]].g = 0
    maze.maze[start_pos[0]][start_pos[1]].f = maze.maze[start_pos[0]][start_pos[1]].h

    # Assigning a random number to start position to the starting position and adding to visited nodes
    node_to_random_number_mapping[start_pos] = 0
    visited_nodes.add(start_pos)

    # Add start position node into the sorted set. We are giving priority to f(n), h(n), and g(n) in the decreasing
    # order. Push random number for random selection if there is conflict between two nodes
    # (If f(n), g(n), and h(n) are same for two nodes)
    sorted_set.add(((maze.maze[start_pos[0]][start_pos[1]].f, maze.maze[start_pos[0]][start_pos[1]].h,
                     maze.maze[start_pos[0]][start_pos[1]].g, node_to_random_number_mapping[start_pos]), start_pos))

    parents[start_pos] = start_pos

    num_explored_nodes = 0

    # Running the loop until we reach our goal state or the sorted set is empty
    while sorted_set.__len__() != 0:
        # Popping first (shortest) element from the sorted set
        current_node = sorted_set.pop(index=0)

        # Increase the number of explored nodes
        num_explored_nodes += 1

        # If we have found the goal position, we can return parents and total explored nodes
        if current_node[1] == goal_pos:
            return parents, num_explored_nodes

        # Otherwise, we need to iterate through each child of the current node
        for val in range(len(X)):
            neighbour = (current_node[1][0] + X[val], current_node[1][1] + Y[val])

            # Neighbour should not go outside our maze and it should not be blocked if we want to visit that particular
            # neighbour
            if check(neighbour, maze.num_cols, maze.num_rows) and (
                    not maze.maze[neighbour[0]][neighbour[1]].is_blocked):

                # If neighbour is being visited first time, we should change its g(n) and f(n) accordingly. Also, we
                # need to assign a random value to it for the time of conflict. In the end, we will add all those things
                # into the sorted set and update its parent
                if neighbour not in visited_nodes:
                    maze.maze[neighbour[0]][neighbour[1]].g = maze.maze[current_node[1][0]][current_node[1][1]].g + 1
                    maze.maze[neighbour[0]][neighbour[1]].f = maze.maze[neighbour[0]][neighbour[1]].g + \
                                                              maze.maze[neighbour[0]][neighbour[1]].h
                    node_to_random_number_mapping[neighbour] = val
                    visited_nodes.add(neighbour)
                    sorted_set.add(((maze.maze[neighbour[0]][neighbour[1]].f, maze.maze[neighbour[0]][neighbour[1]].h,
                                     maze.maze[neighbour[0]][neighbour[1]].g, val), neighbour))
                    parents[neighbour] = current_node[1]

                # If a particular neighbour is already visited, we should compare its f(n) value to its previous f(n)
                # value. If current computed f(n) value is less than the previously computed value, we should remove
                # previously computed value and add new value to the sorted set
                else:
                    neighbour_g = maze.maze[current_node[1][0]][current_node[1][1]].g + 1
                    neighbour_f = maze.maze[neighbour[0]][neighbour[1]].h + neighbour_g
                    if neighbour_f < maze.maze[neighbour[0]][neighbour[1]].f:

                        # The following if condition is needed only when the heuristic is inadmissible otherwise a
                        # neighbour has to be in the sorted set if we are able to find out less value of f(n) for that
                        # particular neighbour
                        if ((maze.maze[neighbour[0]][neighbour[1]].f, maze.maze[neighbour[0]][neighbour[1]].h,
                             maze.maze[neighbour[0]][neighbour[1]].g, node_to_random_number_mapping[neighbour]),
                            neighbour) in sorted_set:
                            sorted_set.remove(
                                ((maze.maze[neighbour[0]][neighbour[1]].f, maze.maze[neighbour[0]][neighbour[1]].h,
                                  maze.maze[neighbour[0]][neighbour[1]].g, node_to_random_number_mapping[neighbour]),
                                 neighbour))
                        maze.maze[neighbour[0]][neighbour[1]].g = neighbour_g
                        maze.maze[neighbour[0]][neighbour[1]].f = neighbour_f
                        node_to_random_number_mapping[neighbour] = val
                        sorted_set.add(
                            ((maze.maze[neighbour[0]][neighbour[1]].f, maze.maze[neighbour[0]][neighbour[1]].h,
                              maze.maze[neighbour[0]][neighbour[1]].g, val), neighbour))
                        parents[neighbour] = current_node[1]

    return parents, num_explored_nodes


def pre_process_input(array: np.array, current_position: tuple, project_no: int = 1, architecture_type: str = 'dense',
                      is_testing: bool = False, sensed: np.array = None):
    if project_no == 1:

        if architecture_type == 'dense':
            array[current_position[0]][current_position[1]] = CURRENT_CELL_WEIGHT
            for ind2 in range(len(X)):
                neighbour = (current_position[0] + X[ind2], current_position[1] + Y[ind2])
                if check(neighbour, TRAINED_MODEL_NUM_ROWS, TRAINED_MODEL_NUM_ROWS):
                    array[neighbour[0]][neighbour[1]] *= NEIGHBOR_WEIGHT
            if is_testing:
                return array.reshape(1, -1)
            else:
                return array.reshape(-1)

        elif architecture_type == 'cnn':
            position = np.zeros((TRAINED_MODEL_NUM_ROWS, TRAINED_MODEL_NUM_COLS))
            position[current_position[0]][current_position[1]] = CURRENT_CELL_WEIGHT
            for ind2 in range(len(X)):
                neighbor = (current_position[0] + X[ind2], current_position[1] + Y[ind2])
                if check(neighbor, TRAINED_MODEL_NUM_ROWS, TRAINED_MODEL_NUM_COLS):
                    position[neighbor[0]][neighbor[1]] = NEIGHBOR_WEIGHT
            if is_testing:
                return np.expand_dims(np.stack(((array % 100) - 1, np.floor(array / 100), position)), axis=0)
            else:
                return np.stack(((array % 100) - 1, np.floor(array / 100), position))

    elif project_no == 2:

        if architecture_type == 'dense':
            array[current_position[0]][current_position[1]] = CURRENT_CELL_WEIGHT
            for ind2 in range(len(X)):
                neighbour = (current_position[0] + X[ind2], current_position[1] + Y[ind2])
                if check(neighbour, TRAINED_MODEL_NUM_ROWS, TRAINED_MODEL_NUM_ROWS):
                    array[neighbour[0]][neighbour[1]] *= NEIGHBOR_WEIGHT
            num_confirmed_blocked = sensed % 10
            num_sensed_blocked = np.floor(sensed / 10) % 10
            num_confirmed_unblocked = np.floor(sensed / 100) % 10
            num_sensed_unblocked = np.floor(sensed / 1000) % 10
            if is_testing:
                return np.stack((array, num_confirmed_blocked, num_sensed_blocked, num_confirmed_unblocked,
                                 num_sensed_unblocked)).reshape(1, -1)
            else:
                return np.stack((array, num_confirmed_blocked, num_sensed_blocked, num_confirmed_unblocked,
                                 num_sensed_unblocked)).reshape(-1)

        elif architecture_type == 'cnn':
            position = np.zeros((TRAINED_MODEL_NUM_ROWS, TRAINED_MODEL_NUM_COLS))
            position[current_position[0]][current_position[1]] = CURRENT_CELL_WEIGHT
            for ind2 in range(len(X)):
                neighbor = (current_position[0] + X[ind2], current_position[1] + Y[ind2])
                if check(neighbor, TRAINED_MODEL_NUM_ROWS, TRAINED_MODEL_NUM_COLS):
                    position[neighbor[0]][neighbor[1]] = NEIGHBOR_WEIGHT

            num_confirmed_blocked = sensed % 10
            num_sensed_blocked = np.floor(sensed / 10) % 10
            num_confirmed_unblocked = np.floor(sensed / 100) % 10
            num_sensed_unblocked = np.floor(sensed / 1000) % 10

            if is_testing:
                return np.expand_dims(np.stack((array, position, num_confirmed_blocked, num_sensed_blocked,
                                                num_confirmed_unblocked, num_sensed_unblocked)), axis=0)
            else:
                return np.stack((array, position, num_confirmed_blocked, num_sensed_blocked, num_confirmed_unblocked,
                                 num_sensed_unblocked))


def explore_neighbors(maze: Maze, maze_array: np.array, cur_pos: tuple, project_no: int = 1,
                      architecture_type: str = 'dense', agent=None):
    if project_no == 1 and architecture_type == 'dense':
        if not maze.maze[cur_pos[0]][cur_pos[1]].is_confirmed:
            maze.maze[cur_pos[0]][cur_pos[1]].is_confirmed = True
            maze.maze_numpy[cur_pos[0]][cur_pos[1]] = UNBLOCKED_WEIGHT * UNBLOCKED_NUMBER

        maze.maze_numpy[cur_pos[0]][cur_pos[1]] += BLOCKED_NUMBER

        for ind in range(len(X)):
            neighbour = (cur_pos[0] + X[ind], cur_pos[1] + Y[ind])
            if check(neighbour, NUM_COLS, NUM_ROWS):
                if not maze.maze[neighbour[0]][neighbour[1]].is_confirmed:
                    maze.maze[neighbour[0]][neighbour[1]].is_confirmed = True
                    if maze_array[neighbour[0]][neighbour[1]] == BLOCKED_NUMBER:
                        maze.maze[neighbour[0]][neighbour[1]].is_blocked = True
                        maze.maze_numpy[neighbour[0]][neighbour[1]] = BLOCKED_NUMBER
                    else:
                        maze.maze_numpy[neighbour[0]][neighbour[1]] = UNBLOCKED_WEIGHT * UNBLOCKED_NUMBER

    elif project_no == 1 and architecture_type == 'cnn':
        maze.maze[cur_pos[0]][cur_pos[1]].is_confirmed = True
        maze.maze_numpy[cur_pos[0]][cur_pos[1]] = UNBLOCKED_NUMBER
        maze.num_times_cell_visited[cur_pos[0]][cur_pos[1]] += 1

        for ind in range(len(X)):
            neighbour = (cur_pos[0] + X[ind], cur_pos[1] + Y[ind])
            if check(neighbour, NUM_COLS, NUM_ROWS):
                if not maze.maze[neighbour[0]][neighbour[1]].is_confirmed:
                    maze.maze[neighbour[0]][neighbour[1]].is_confirmed = True
                    if maze_array[neighbour[0]][neighbour[1]] == BLOCKED_NUMBER:
                        maze.maze[neighbour[0]][neighbour[1]].is_blocked = True
                        maze.maze_numpy[neighbour[0]][neighbour[1]] = BLOCKED_NUMBER
                    else:
                        maze.maze_numpy[neighbour[0]][neighbour[1]] = UNBLOCKED_NUMBER

    elif project_no == 2:
        maze.maze[cur_pos[0]][cur_pos[1]].is_confirmed = True
        maze.maze_numpy[cur_pos[0]][cur_pos[1]] = UNBLOCKED_NUMBER
        maze.num_times_cell_visited[cur_pos[0]][cur_pos[1]] += 1

        if agent.maze[cur_pos[0]][cur_pos[1]].is_visited:
            agent.maze[cur_pos[0]][cur_pos[1]].is_visited = True
            sense_current_node(agent.maze, cur_pos, maze_array,
                               num_confirmed_blocked=agent.num_confirmed_blocked,
                               num_sensed_blocked=agent.num_sensed_blocked,
                               num_confirmed_unblocked=agent.num_confirmed_unblocked,
                               num_sensed_unblocked=agent.num_sensed_unblocked)
            # full_maze[children[current_position][0]][children[current_position][1]] == BLOCKED_NUMBER:
        find_block_while_inference(agent.maze, cur_pos, maze_array, maze.maze_numpy, agent.num_confirmed_blocked,
                                   agent.num_confirmed_unblocked)


def repeated_forward(maze: Maze, maze_array: np.array, data: list, start_pos: tuple, goal_pos: tuple,
                     is_field_of_view_explored: bool = True, project_no: int = 1, architecture_type: str = 'dense'):
    """
    This is the repeated forward function which can be used with any algorithm (astar or bfs). This function will
    repeatedly call corresponding algorithm function until it reaches goal or finds out there is no path till goal.
    :param maze: Maze array of agent
    :param maze_array: Original (Full) Maze array
    :param start_pos: starting position of the maze from where agent want to start
    :param goal_pos: goal state where agent want to reach
    :param is_field_of_view_explored: It will explore field of view if this attribute is true otherwise it won't.
    :return: This function will return final paths on which agent moved to reach goal or empty list if agent can't find
            path to goal. Second is total number of processed nodes while running the algorithm.
    """

    # defining the following two attributes to find which would be useful to return values
    final_paths = list()

    if is_field_of_view_explored:
        explore_neighbors(maze, maze_array, start_pos, project_no, architecture_type)

    # Running the while loop until we will get a path from start_pos to goal_pos or we have figured out there is no path
    # from start_pos to goal_pos
    while True:

        parents, num_explored_nodes = astar_search(maze, start_pos, goal_pos)

        # If goal_pos doesn't exist in parents which means path is not available so returning empty list.
        if goal_pos not in parents:
            # if data is not None:
            #     if project_no == 1:
            #         if architecture_type == 'dense':
            #             data.append({
            #                 'current_pos': start_pos,
            #                 'input': maze.maze_numpy.copy(),
            #                 'output': TARGET_CANNOT_BE_REACHED_NUMBER
            #             })
            #         elif architecture_type == 'cnn':
            #             data.append({
            #                 'current_pos': start_pos,
            #                 'input': maze.maze_numpy.copy() + 100 * maze.num_times_cell_visited.copy() + 1,
            #                 # 'num_times_cell_visited': maze.num_times_cell_visited.copy(),
            #                 'output': TARGET_CANNOT_BE_REACHED_NUMBER
            #             })
            #         else:
            #             raise Exception("Architecture must be dense or cnn")

            return final_paths, 0

        # parents contains parent of each node through path from start_pos to goal_pos. To store path from start_pos to
        # goal_pos, we need to store child of each node starting from start_pos.
        cur_pos = goal_pos
        children = dict()

        children[cur_pos] = cur_pos

        # Storing child of each node so we can iterate from start_pos to goal_pos
        while cur_pos != parents[cur_pos]:
            children[parents[cur_pos]] = cur_pos
            cur_pos = parents[cur_pos]

        # Setting current position to starting position so we can start iterating from start_pos
        cur_pos = start_pos

        current_path = [cur_pos]

        # Iterating from start_pos to goal_pos if we won't get any blocks in between otherwise we are terminating the
        # iteration.
        while cur_pos != children[cur_pos]:

            # maze.maze_numpy[cur_pos[0]][cur_pos[1]] = UNBLOCKED_NUMBER

            # Explore the field of view and update the blocked nodes if there's any in the path.
            if is_field_of_view_explored and start_pos != cur_pos:
                explore_neighbors(maze, maze_array, cur_pos, project_no, architecture_type)

            # If we encounter any block in the path, we have to terminate the iteration
            if maze_array[children[cur_pos][0]][children[cur_pos][1]] == BLOCKED_NUMBER:
                break

            if data is not None:

                if (project_no == 1) and (architecture_type == 'dense'):
                    data.append({
                        'current_pos': cur_pos,
                        'input': maze.maze_numpy.copy(),
                        'output': find_output(cur_pos, children[cur_pos])
                    })

                if (project_no == 1) and (architecture_type == 'cnn'):
                    data.append({
                        'current_pos': cur_pos,
                        'input': maze.maze_numpy.copy() + 100 * maze.num_times_cell_visited.copy() + 1,
                        # 'num_times_cell_visited': maze.num_times_cell_visited.copy(),
                        'output': find_output(cur_pos, children[cur_pos])
                    })

            cur_pos = children[cur_pos]
            current_path.append(cur_pos)

        # If we are able to find the goal state, we should return the path and total explored nodes.
        if cur_pos == goal_pos:
            final_paths.append(current_path)
            return final_paths, 1
        else:

            # Change the start node to last unblocked node and backtrack if it is set to any positive integer.
            maze.maze[children[cur_pos][0]][children[cur_pos][1]].is_blocked = True
            maze.maze_numpy[children[cur_pos][0]][children[cur_pos][1]] = BLOCKED_NUMBER
            cur_pos = current_path[-1]

            final_paths.append(current_path)
            start_pos = cur_pos


def bootstraping(maze: Maze, current_position: tuple, num_rows: int, num_columns: int, num_samples: int):
    x_start = max(0, current_position[0] - TRAINED_MODEL_NUM_ROWS + 1)
    x_end = min(current_position[0], max(num_rows - TRAINED_MODEL_NUM_ROWS, 0))

    y_start = max(0, current_position[1] - TRAINED_MODEL_NUM_COLS + 1)
    y_end = min(current_position[1], max(num_columns - TRAINED_MODEL_NUM_COLS, 0))

    starting_positions = list()
    for ind in range(num_samples):
        x, y = random.randint(x_start, x_end), random.randint(y_start, y_end)
        while maze.maze[x][y].is_blocked or \
                maze.maze[x + TRAINED_MODEL_NUM_ROWS - 1][y + TRAINED_MODEL_NUM_COLS - 1].is_blocked:
            x, y = random.randint(x_start, x_end), random.randint(y_start, y_end)
        starting_positions.append((x, y))
    return starting_positions


def make_action(maze: Maze, model, current_position: tuple, num_samples: int, project_no: int = 1,
                architecture_type='dense', sensed: np.array = None):
    start_positions = bootstraping(maze, current_position, NUM_ROWS, NUM_COLS, num_samples)
    action = np.zeros(4)
    for start_pos in start_positions:
        if project_no == 1:
            if architecture_type == 'dense':
                array = maze.maze_numpy[start_pos[0]: start_pos[0] + TRAINED_MODEL_NUM_ROWS,
                        start_pos[1]: start_pos[1] + TRAINED_MODEL_NUM_COLS].copy()
            elif architecture_type == 'cnn':
                array = maze.maze_numpy[start_pos[0]: start_pos[0] + TRAINED_MODEL_NUM_ROWS,
                        start_pos[1]: start_pos[1] + TRAINED_MODEL_NUM_COLS].copy() + \
                        100 * maze.num_times_cell_visited[start_pos[0]: start_pos[0] + TRAINED_MODEL_NUM_ROWS,
                              start_pos[1]: start_pos[1] + TRAINED_MODEL_NUM_COLS].copy() + 1
            else:
                array = None
        else:
            num_times_cell_visited = maze.num_times_cell_visited[start_pos[0]: start_pos[0] + TRAINED_MODEL_NUM_ROWS,
                        start_pos[1]: start_pos[1] + TRAINED_MODEL_NUM_COLS].copy()
            maze_numpy = maze.maze_numpy[start_pos[0]: start_pos[0] + TRAINED_MODEL_NUM_ROWS,
                        start_pos[1]: start_pos[1] + TRAINED_MODEL_NUM_COLS].copy()
            maze_numpy[maze_numpy == UNBLOCKED_NUMBER] *= UNBLOCKED_WEIGHT
            array = maze_numpy - num_times_cell_visited
        action += np.array(model.predict(pre_process_input(array, (current_position[0] - start_pos[0],
                                                                   current_position[1] - start_pos[1]),
                                                           project_no=project_no,
                                                           architecture_type=architecture_type, is_testing=True,
                                                           sensed=sensed[start_pos[0]: start_pos[0] +
                                                                                       TRAINED_MODEL_NUM_ROWS,
                                                                  start_pos[1]: start_pos[1] + TRAINED_MODEL_NUM_COLS]
                                                           .copy()))[0])
    return np.argmax(action)


def repeated_forward_astar(maze_array: np.array, start_pos: tuple, goal_pos: tuple, model1, model2=None,
                           is_field_of_view_explored: bool = True):
    final_paths = list()
    total_actions = 0
    dense_valid_actions = 0
    cnn_valid_actions = 0

    mazes = [Maze(NUM_ROWS, NUM_COLS), Maze(NUM_ROWS, NUM_COLS)]

    if is_field_of_view_explored:
        explore_neighbors(mazes[0], maze_array, start_pos, 1, 'dense')
        if model2 is not None:
            explore_neighbors(mazes[1], maze_array, start_pos, 1, 'cnn')

    # Running the while loop until we will get a path from start_pos to goal_pos or we have figured out there is no path
    # from start_pos to goal_pos
    while True:

        parents = astar_search(mazes[0], start_pos, goal_pos)[0]

        # If goal_pos doesn't exist in parents which means path is not available so returning empty list.
        if goal_pos not in parents:
            action1 = make_action(mazes[0], model1, start_pos, 1, 1, 'dense')
            if model2 is not None:
                action2 = make_action(mazes[1], model2, start_pos, 1, 1, 'cnn')

            output1 = 1
            output2 = 1

            if action1 == TARGET_CANNOT_BE_REACHED_NUMBER:
                output1 = 0
            if model2 is not None and action2 == TARGET_CANNOT_BE_REACHED_NUMBER:
                output2 = 0

            return total_actions, dense_valid_actions, cnn_valid_actions, 0, output1, output2
        # parents contains parent of each node through path from start_pos to goal_pos. To store path from start_pos to
        # goal_pos, we need to store child of each node starting from start_pos.
        cur_pos = goal_pos
        children = dict()

        children[cur_pos] = cur_pos

        # Storing child of each node so we can iterate from start_pos to goal_pos
        while cur_pos != parents[cur_pos]:
            children[parents[cur_pos]] = cur_pos
            cur_pos = parents[cur_pos]

        # Setting current position to starting position so we can start iterating from start_pos
        cur_pos = start_pos

        current_path = [cur_pos]

        # Iterating from start_pos to goal_pos if we won't get any blocks in between otherwise we are terminating the
        # iteration.
        while cur_pos != children[cur_pos]:

            # maze.maze_numpy[cur_pos[0]][cur_pos[1]] = UNBLOCKED_NUMBER

            # Explore the field of view and update the blocked nodes if there's any in the path.
            if is_field_of_view_explored and start_pos != cur_pos:
                explore_neighbors(mazes[0], maze_array, cur_pos, 1, 'dense')
                if model2 is not None:
                    explore_neighbors(mazes[1], maze_array, cur_pos, 1, 'cnn')

            # If we encounter any block in the path, we have to terminate the iteration
            if maze_array[children[cur_pos][0]][children[cur_pos][1]] == BLOCKED_NUMBER:
                break

            action1 = make_action(mazes[0], model1, start_pos, 1, 1, 'dense')
            if model2 is not None:
                action2 = make_action(mazes[1], model2, start_pos, 1, 1, 'cnn')

            total_actions += 1
            if action1 == find_output(cur_pos, children[cur_pos]):
                dense_valid_actions += 1
            if model2 is not None and action2 == find_output(cur_pos, children[cur_pos]):
                cnn_valid_actions += 1
            cur_pos = children[cur_pos]
            current_path.append(cur_pos)

        # If we are able to find the goal state, we should return the path and total explored nodes.
        if cur_pos == goal_pos:
            final_paths.append(current_path)
            return total_actions, dense_valid_actions, cnn_valid_actions, 1, 1, 1
        else:

            # Change the start node to last unblocked node and backtrack if it is set to any positive integer.
            mazes[0].maze[children[cur_pos][0]][children[cur_pos][1]].is_blocked = True
            mazes[0].maze_numpy[children[cur_pos][0]][children[cur_pos][1]] = BLOCKED_NUMBER

            mazes[1].maze[children[cur_pos][0]][children[cur_pos][1]].is_blocked = True
            mazes[1].maze_numpy[children[cur_pos][0]][children[cur_pos][1]] = BLOCKED_NUMBER
            cur_pos = current_path[-1]

            final_paths.append(current_path)
            start_pos = cur_pos


def generate_random_position(maze: Maze, current_position: tuple):
    possible_actions = list()
    for ind in range(len(X)):
        neighbor = (current_position[0] + X[ind], current_position[1] + Y[ind])
        if check(neighbor, NUM_ROWS, NUM_COLS):
            if not maze.maze[neighbor[0]][neighbor[1]].is_blocked:
                possible_actions.append(ind)
    if len(possible_actions) > 0:
        action = random.choice(possible_actions)
        return current_position[0] + X[action], current_position[1] + Y[action]
    else:
        return current_position


def ml_agent_dfs(maze: Maze, full_maze: np.array, start_position: tuple, goal_position: tuple, project_no: int = 1,
                 architecture_type: str = 'dense', model=None, agent=None):
    current_position = start_position
    num_samples = 2
    trajectory_length = 0
    predicted_blocked_cells = 0
    predicted_out_of_maze_cells = 0

    previous_prediction = (-4, -4)
    previous_frequency = 0
    while True:
        # Exploration
        if trajectory_length > TRAJECTORY_LENGTH_THRESHOLD:
            return trajectory_length, 0, predicted_blocked_cells, predicted_out_of_maze_cells
        explore_neighbors(maze, full_maze, current_position, project_no=project_no, architecture_type=architecture_type,
                          agent=agent)
        if agent is not None:
            sensed = 1 * agent.num_confirmed_blocked + 10 * agent.num_sensed_blocked + \
                     100 * agent.num_confirmed_unblocked + 1000 * agent.num_sensed_unblocked
        else:
            sensed = None
        action = make_action(maze, model, current_position, num_samples, project_no=project_no,
                             architecture_type=architecture_type, sensed=sensed)

        # if action == TARGET_CANNOT_BE_REACHED_NUMBER:
        #     print(maze.maze_numpy)
        #     return trajectory_length, 0
        next_position = (current_position[0] + X[action], current_position[1] + Y[action])
        trajectory_length += 1
        if check(next_position, NUM_COLS, NUM_ROWS):
            if full_maze[next_position[0]][next_position[1]] == BLOCKED_NUMBER:

                if maze.maze_numpy[next_position[0]][next_position[1]] <= BLOCKED_NUMBER:
                    maze.maze_numpy[next_position[0]][next_position[1]] += BLOCKED_NUMBER
                    # print('blocked cell')
                    # print(current_position)
                    # print(next_position)

                    predicted_blocked_cells += 1

                    if previous_prediction == next_position:
                        previous_frequency += 1
                    else:
                        previous_prediction = next_position
                        previous_frequency = 1

                    if previous_frequency > 5:
                        next_position = generate_random_position(maze, current_position)
                        if next_position == current_position:
                            return trajectory_length, 0, predicted_blocked_cells, predicted_out_of_maze_cells
                        else:
                            current_position = next_position
                        previous_prediction = (-4, -4)
                        previous_frequency = 0
                else:
                    maze.maze_numpy[next_position[0]][next_position[1]] = BLOCKED_NUMBER
                    maze.maze[next_position[0]][next_position[1]].is_blocked = True
                    maze.maze[next_position[0]][next_position[1]].is_confimed = True
            else:
                current_position = next_position
                previous_prediction = (-4, -4)
                previous_frequency = 0
        else:
            # print("out of bound")
            # print(current_position)
            # print(next_position)

            predicted_out_of_maze_cells += 1

            for ind in range(len(X)):
                neighbor = (current_position[0] + X[ind], current_position[1] + Y[ind])
                if check(neighbor, NUM_ROWS, NUM_COLS):
                    if not maze.maze[neighbor[0]][neighbor[1]].is_blocked:
                        maze.maze_numpy[neighbor[0]][neighbor[1]] += UNBLOCKED_NUMBER

            if previous_prediction == next_position:
                previous_frequency += 1
            else:
                previous_prediction = next_position
                previous_frequency = 1
            if previous_frequency > 5:
                next_position = generate_random_position(maze, current_position)
                if next_position == current_position:
                    return trajectory_length, 0, predicted_blocked_cells, predicted_out_of_maze_cells
                else:
                    current_position = next_position
                previous_prediction = (-4, -4)
                previous_frequency = 0
            # return -1, 0

        if current_position == goal_position:
            print('Reached goal position')
            return trajectory_length, 1, predicted_blocked_cells, predicted_out_of_maze_cells


def parent_to_child_dict(parent: dict, starting_position: tuple):
    """
    This function is helpful to generate children dictionary from parents dictionary
    :param parent: parent dictionary
    :param starting_position: starting position of the last function
    :return: generate child dictionary from parent.
    """
    child = dict()

    child[starting_position] = starting_position
    cur_pos = starting_position

    # Storing child of each node so we can iterate from start_pos to goal_pos
    while cur_pos != parent[cur_pos]:
        child[parent[cur_pos]] = cur_pos
        cur_pos = parent[cur_pos]

    return child


def sense_current_node(maze, current_position: tuple, full_maze: np.array, num_confirmed_blocked: np.array,
                       num_sensed_blocked: np.array, num_confirmed_unblocked: np.array, num_sensed_unblocked: np.array):
    """
    This function is used to sense the current node and update details in the knowledge base
    :param maze: Maze object
    :param current_position: position of the maze for which you want to sense
    :param full_maze: full maze
    :return: Nothing as we are directly updating into maze object
    """

    # Visit all eight neighbors and sense them
    for neighbor in maze[current_position[0]][current_position[1]].eight_neighbors:

        # Update the status for confirm and non-confirm conditions
        if maze[neighbor[0]][neighbor[1]].is_confirmed:
            if maze[neighbor[0]][neighbor[1]].is_blocked:
                maze[current_position[0]][current_position[1]].num_confirmed_blocked += 1
                maze[current_position[0]][current_position[1]].num_sensed_blocked += 1

                num_confirmed_blocked[current_position[0]][current_position[1]] += 1
                num_sensed_blocked[current_position[0]][current_position[1]] += 1
            else:
                maze[current_position[0]][current_position[1]].num_confirmed_unblocked += 1
                maze[current_position[0]][current_position[1]].num_sensed_unblocked += 1

                num_confirmed_unblocked[current_position[0]][current_position[1]] += 1
                num_sensed_unblocked[current_position[0]][current_position[1]] += 1
        else:

            if full_maze[neighbor[0]][neighbor[1]] == BLOCKED_NUMBER:
                maze[current_position[0]][current_position[1]].num_sensed_blocked += 1
                num_sensed_blocked[current_position[0]][current_position[1]] += 1
            else:
                maze[current_position[0]][current_position[1]].num_sensed_unblocked += 1
                num_sensed_unblocked[current_position[0]][current_position[1]] += 1


def can_infer(num_sensed_blocked: int, num_confirmed_blocked: int, num_sensed_unblocked: int,
              num_confirmed_unblocked: int):
    """
    check whether we can infer or not from the current set of variables
    :param num_sensed_blocked: number of sensed blocks
    :param num_confirmed_blocked: number of confirmed blocks
    :param num_sensed_unblocked: number of sensed unblocks
    :param num_confirmed_unblocked: number confirmed unblocks
    :return: True if we can infer anything otherwise False
    """

    # Check precondition
    assert (num_sensed_blocked >= num_confirmed_blocked) and (num_sensed_unblocked >= num_confirmed_unblocked)

    # Condition whether we can infer or not
    if ((num_sensed_blocked == num_confirmed_blocked) and (num_sensed_unblocked > num_confirmed_unblocked)) or \
            ((num_sensed_unblocked == num_confirmed_unblocked) and (num_sensed_blocked > num_confirmed_blocked)):
        return True
    return False


def find_block_while_inference(maze: list, current_position: tuple, full_maze: np.array, maze_numpy: np.array,
                               num_confirmed_blocked: np.array, num_confirmed_unblocked: np.array,
                               entire_trajectory_nodes=None):
    """
    This function helps us find blocks in the path if they exist while also performing inference.
    :param maze: This agents copy of the maze.
    :param current_position: This is the current position of the agent.
    :param full_maze: This is the full maze.
    :param entire_trajectory_nodes: This is the set of nodes that are in the agents current trajectory.
    :return: True if there is a block in path else false.
    """

    # We create this queue to help us store all the variables that are going to be confirmed and subsequently store
    # those cells in a set so that we can access them in constant time.
    if entire_trajectory_nodes is None:
        entire_trajectory_nodes = list()
    inference_items = Queue()
    items_in_the_queue = set()
    is_block_node_in_current_path = False

    # We put the current position of the agent into the Queue and set.
    inference_items.put(current_position)
    items_in_the_queue.add(current_position)

    # We run this while loop until the queue is empty and there is nothing left to be inferred.
    while not inference_items.empty():
        current_node = inference_items.get()
        items_in_the_queue.remove(current_node)

        # If the current node is not set as confirmed in the maze then we set it to confirmed and made updates
        # accordingly
        if not maze[current_node[0]][current_node[1]].is_confirmed:

            maze[current_node[0]][current_node[1]].is_confirmed = True

            if full_maze[current_node[0]][current_node[1]] == BLOCKED_NUMBER:
                maze[current_node[0]][current_node[1]].is_blocked = True

                if current_node == current_position:
                    maze_numpy[current_node[0]][current_node[1]] = BLOCKED_NUMBER

                # If current node is in the trajectory then we should return True for this function
                if current_node in entire_trajectory_nodes:
                    is_block_node_in_current_path = True
            else:
                maze[current_node[0]][current_node[1]].is_blocked = False

                if current_node == current_position:
                    maze_numpy[current_node[0]][current_node[1]] = UNBLOCKED_NUMBER

            # Iterate over eight neighbors, update their status and add it into the queue
            for neighbor in maze[current_node[0]][current_node[1]].eight_neighbors:
                if not maze[neighbor[0]][neighbor[1]].is_visited:
                    continue
                if maze[current_node[0]][current_node[1]].is_blocked:
                    maze[neighbor[0]][neighbor[1]].num_confirmed_blocked += 1

                    if current_node == current_position:
                        num_confirmed_blocked[neighbor[0]][neighbor[1]] += 1
                else:
                    maze[neighbor[0]][neighbor[1]].num_confirmed_unblocked += 1

                    if current_node == current_position:
                        num_confirmed_unblocked[neighbor[0]][neighbor[1]] += 1

                if not (neighbor in items_in_the_queue):
                    items_in_the_queue.add(neighbor)
                    inference_items.put(neighbor)

        # If the current cell is visited, we can inference rules to them
        if maze[current_node[0]][current_node[1]].is_visited:

            # This rule applies to the 3rd as well as the 4th agent where we are trying to infer using only one
            # cell's inference
            if can_infer(maze[current_node[0]][current_node[1]].num_sensed_blocked,
                         maze[current_node[0]][current_node[1]].num_confirmed_blocked,
                         maze[current_node[0]][current_node[1]].num_sensed_unblocked,
                         maze[current_node[0]][current_node[1]].num_confirmed_unblocked):

                for neighbor in maze[current_node[0]][current_node[1]].eight_neighbors:
                    if (neighbor not in items_in_the_queue) and \
                            (not maze[neighbor[0]][neighbor[1]].is_confirmed):
                        items_in_the_queue.add(neighbor)
                        inference_items.put(neighbor)

    return is_block_node_in_current_path

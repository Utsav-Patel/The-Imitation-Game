import numpy as np

from src.Node import Node


class Maze:
    def __init__(self, num_cols: int, num_rows: int):
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.maze_numpy = np.zeros((num_rows, num_cols))
        self.maze = list()
        for row_num in range(self.num_rows):
            lst = list()
            for column_num in range(self.num_cols):
                lst.append(Node())
            self.maze.append(lst)

    def __str__(self):
        return 'NUmber of columns: ' + str(self.num_cols) + '\nNumber of rows: ' + str(self.num_rows) \
               + '\nMaze: ' + str(self.maze)

    def reset(self):
        self.maze_numpy.fill(0)
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                self.maze[row][col].reset()

    def reset_except_h(self):
        self.maze_numpy.fill(0)
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                self.maze[row][col].reset_except_h()
NUM_COLS = 20
NUM_ROWS = 20
INF = 1e9

CHECKPOINT_FILEPATH = "../checkpoints/project1_dense/20x20/added_neighbors_added_num_visited_8000/cp-{epoch:04d}.ckpt"
DATA_PATH = "../data/project1_dense/20x20/added_neighbors_num_visited_multiprocessing_8000.pkl"

STARTING_POSITION_OF_AGENT = (0, 0)
GOAL_POSITION_OF_AGENT = (NUM_ROWS - 1, NUM_COLS - 1)

X = [-1, 0, 1, 0]
Y = [0, 1, 0, -1]

UNVISITED_NUMBER = 0
UNBLOCKED_NUMBER = 3
BLOCKED_NUMBER = -1
TARGET_CANNOT_BE_REACHED_NUMBER = 4

NEIGHBOR_WEIGHT = 25
CURRENT_CELL_WEIGHT = 100

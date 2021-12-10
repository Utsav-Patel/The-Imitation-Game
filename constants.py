import os

PROJECT_PATH = os.path.dirname(__file__)
PROJECT_NO = 1
ARCHITECTURE_TYPE = 'cnn'

NUM_COLS = 20
NUM_ROWS = 20
INF = 1e9

FILE_PREFIX = "20x20"
CHECKPOINT_FILEPATH = os.path.join(PROJECT_PATH, "checkpoints", "project" + str(PROJECT_NO), ARCHITECTURE_TYPE,
                                   FILE_PREFIX + "-{epoch:04d}.ckpt")
DATA_PATH = os.path.join(PROJECT_PATH, "data", "project" + str(PROJECT_NO), ARCHITECTURE_TYPE, FILE_PREFIX + ".pkl")

STATE_OF_THE_ART_MODEL_PROJECT1_DENSE_CHECKPOINT_PATH = os.path.join(PROJECT_PATH, "checkpoints", "project1", "dense")
STATE_OF_THE_ART_MODEL_PROJECT1_CNN_CHECKPOINT_PATH = os.path.join(PROJECT_PATH, "checkpoints", "project1", "cnn")

STARTING_POSITION_OF_AGENT = (0, 0)
GOAL_POSITION_OF_AGENT = (NUM_ROWS - 1, NUM_COLS - 1)

X = [-1, 0, 1, 0]
Y = [0, 1, 0, -1]

UNVISITED_NUMBER = 0
BLOCKED_NUMBER = -1
UNBLOCKED_NUMBER = 1
TARGET_CANNOT_BE_REACHED_NUMBER = 4

# if PROJECT_NO == 1:
#     if ARCHITECTURE_TYPE == 'dense':
#         UNBLOCKED_NUMBER = 3
#     elif ARCHITECTURE_TYPE == 'cnn':
#         UNBLOCKED_NUMBER = 1
#     else:
#         raise Exception("Architecture type must be dense or cnn")


NEIGHBOR_WEIGHT = 25
CURRENT_CELL_WEIGHT = 100

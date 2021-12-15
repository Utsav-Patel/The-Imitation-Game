import os

PROJECT_PATH = os.path.dirname(__file__)
PROJECT_NO = 2
ARCHITECTURE_TYPE = 'dense'

NUM_COLS = 50
NUM_ROWS = 50

TRAINED_MODEL_NUM_ROWS = 20
TRAINED_MODEL_NUM_COLS = 20
INF = 1e9

FILE_PREFIX = "20x20"
FILE_SUFFIX = "new"
TRAIN_DATA_PREFIX = "10_to_35_probability_and_7000_each"
VALIDATION_TEST_DATA_PREFIX = "validation_plus_test"

CHECKPOINT_FILEPATH = os.path.join(PROJECT_PATH, "checkpoints", "project" + str(PROJECT_NO), ARCHITECTURE_TYPE,
                                   FILE_PREFIX, FILE_SUFFIX, FILE_SUFFIX + "-{epoch:04d}.ckpt")

DATA_PATH = os.path.join(PROJECT_PATH, "data", "project" + str(PROJECT_NO), ARCHITECTURE_TYPE, FILE_PREFIX,
                         TRAIN_DATA_PREFIX + ".pkl")
VALIDATION_TEST_PATH = os.path.join(PROJECT_PATH, "data", "project" + str(PROJECT_NO), ARCHITECTURE_TYPE, FILE_PREFIX,
                                    VALIDATION_TEST_DATA_PREFIX + ".pkl")

STATE_OF_THE_ART_MODEL_PROJECT1_DENSE_CHECKPOINT_PATH = os.path.join(PROJECT_PATH, "checkpoints", "project1", "dense",
                                                                     FILE_PREFIX, FILE_SUFFIX)
STATE_OF_THE_ART_MODEL_PROJECT1_CNN_CHECKPOINT_PATH = os.path.join(PROJECT_PATH, "checkpoints", "project1", "cnn",
                                                                   FILE_PREFIX, FILE_SUFFIX)
STATE_OF_THE_ART_MODEL_PROJECT2_DENSE_CHECKPOINT_PATH = os.path.join(PROJECT_PATH, "checkpoints", "project2", "dense",
                                                                     FILE_PREFIX, FILE_SUFFIX)
STATE_OF_THE_ART_MODEL_PROJECT2_CNN_CHECKPOINT_PATH = os.path.join(PROJECT_PATH, "checkpoints", "project2", "cnn",
                                                                   FILE_PREFIX, FILE_SUFFIX)

PROJECT2_DATA_PATH = os.path.join(PROJECT_PATH, "data", "project" + str(PROJECT_NO), FILE_PREFIX,
                                  TRAIN_DATA_PREFIX + ".pkl")
PROJECT2_VALIDATION_PATH = os.path.join(PROJECT_PATH, "data", "project" + str(PROJECT_NO), FILE_PREFIX,
                                        VALIDATION_TEST_DATA_PREFIX + ".pkl")

STARTING_POSITION_OF_AGENT = (0, 0)
GOAL_POSITION_OF_AGENT = (NUM_ROWS - 1, NUM_COLS - 1)

X = [-1, 0, 1, 0]
Y = [0, 1, 0, -1]

UNVISITED_NUMBER = 0
BLOCKED_NUMBER = -1
UNBLOCKED_NUMBER = 1
TARGET_CANNOT_BE_REACHED_NUMBER = 4
UNBLOCKED_WEIGHT = 5

NEIGHBOR_WEIGHT = 10
CURRENT_CELL_WEIGHT = 100
TRAJECTORY_LENGTH_THRESHOLD = 500

FLAT_FALSE_NEGATIVE_RATE = 0.2
HILLY_FALSE_NEGATIVE_RATE = 0.5
FOREST_FALSE_NEGATIVE_RATE = 0.8

ZERO_PROBABILITY = 0.0
ONE_PROBABILITY = 1.0

NUM_ITERATIONS = 1

PROBABILITY_OF_GRID = 0.3

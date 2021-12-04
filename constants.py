NUM_COLS = 10
NUM_ROWS = 10
INF = 1e9

CHECKPOINT_FILEPATH = "../checkpoints/training_10x10_model_added_unvisited/cp-{epoch:04d}.ckpt"
DATA_PATH = "../data/10x10/added_nonvisited_change_from_0_to_5.pkl"

STARTING_POSITION_OF_AGENT = (0, 0)
GOAL_POSITION_OF_AGENT = (NUM_ROWS - 1, NUM_COLS - 1)

X = [-1, 0, 1, 0]
Y = [0, 1, 0, -1]

# Hyperparameters for DQN agent, memory and training
EPISODES = 10_000 #3500
HEIGHT = 84
WIDTH = 84
HISTORY_SIZE = 4
TRAINABLE_ACTIONS = [0, 2, 3]
EXPLORE_STEPS = 300_000 #orignally 500000, lowered because only training every 4th step
EPSILON_MIN = 0.1  # was 0.1 for std runs
TRAINING_STEPS = 500_000
train_frame = 50_000 #Originally 100000.  Lower to speed up testing during debugging
sticky_action_prob = 0.05  #for some reason this introduces random ball launch direction, even though it mildly harms training by making ALL actions ocassionally repeat
learning_rate = 0.0005  #was 0.0001 for standardized test 12 and 13

evaluation_reward_length = 50  #changed from 100
Memory_capacity = 1_000_000 #Originally 1000000

BATCH_SIZE = 64 #Originally 128
scheduler_gamma = 0.4  #was 0.65 for standardized test 12 and 13
scheduler_step_size = 100_000  #orignally 100_000

update_target_network_frequency = 5_000 # was 10_000 for standardized test 12 and 13

# Hyperparameters for DQN LSTM agent
lstm_seq_length = 5

#PER Hyperparameters
PER_ALPHA = 0.6
IS_BETA = 0.4
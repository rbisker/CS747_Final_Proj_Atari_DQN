import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import *
from model import DQN, DQN_LSTM
from utils import *
from config import *
import os
import pickle
import math
from torch.amp import autocast, GradScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size, replay_buffer_path=None):
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_max = self.epsilon
        self.epsilon_min = EPSILON_MIN
        self.explore_step = EXPLORE_STEPS 
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.epsilon_decay_rate = 0.0001
        self.train_start = train_frame 
        self.update_target = update_target_network_frequency
        self.beta = self.beta_min = IS_BETA
        self.alpha = self.alpha_max = PER_ALPHA
        self.scaler = GradScaler('cuda')  # for NVIDIA automatic mixed precision
        print(self.get_agent_type() + " initialized")

        #Load replay buffer from file if a path is provided
        if replay_buffer_path and os.path.exists(replay_buffer_path):
            with open(replay_buffer_path, 'rb') as f:
                self.memory = pickle.load(f)
            print(f"Replay buffer loaded from '{replay_buffer_path}' with size {len(self.memory)}")
        else:
            if replay_buffer_path:
                self.policy_net.eval() # Ensure inference mode
                print(f"Replay buffer path '{replay_buffer_path}' not found. Creating new empty buffer.")
            else:
                print("No replay buffer path provided. Creating new empty standard replay buffer.")
                self.memory = ReplayMemory()

        # print("Setting up Circular PER Replay Buffer of size", Memory_capacity)
        # self.memory = CircularReplayMemoryPER(Memory_capacity, HISTORY_SIZE)
        # self.memory = ReplayMemory()

        # Create the policy and target nets and sync them
        self.policy_net = DQN(action_size).to(device)
        self.target_net = DQN(action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        
    def get_agent_type(self):
        return "DQN Agent"

    def load_policy_net(self, path):
        # self.policy_net = torch.load(path)
        self.policy_net.load_state_dict(torch.load(path))

    def update_target_net(self):
        # Update the target network with the current policy network's weights
        self.target_net.load_state_dict(self.policy_net.state_dict())

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            # Choose a random action
            a = np.random.choice(self.action_size)
        else:
            # Choose the best action
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)  # Add batch dimension
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            a = torch.argmax(q_values, dim=1).item()
        return int(a)


    def train_policy_net(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            # self.epsilon = self.epsilon_max / (1 + self.epsilon_decay * frame)
            # Clamp in case of overshoot
            self.epsilon = max(self.epsilon, self.epsilon_min)
            
        # Sample and unpack
        if type(self.memory) == ReplayMemory:
            mini_batch = self.memory.improved_sample_mini_batch(BATCH_SIZE)
            use_per = False
        elif type(self.memory) == CircularReplayMemoryPER:
            mini_batch, mini_batch_indices = self.memory.improved_sample_mini_batch(BATCH_SIZE, per_alpha=self.alpha)
            use_per = True
        else:
            raise NotImplementedError("Invalid replay buffer type")
        
    
        mini_batch = np.array(mini_batch, dtype=object).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :HISTORY_SIZE, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        terminations = list(mini_batch[3]) # checks if the game round is over
        terminations = torch.tensor(terminations, dtype=torch.bool).to(device)

        
        self.policy_net.train()  
        with autocast(device_type='cuda'):
            # Compute Q(s_t, a), the Q-value of the current state
            q_values_all = self.policy_net(states)
            q_values = torch.gather(q_values_all, dim=1, index=actions.unsqueeze(1))

            #Monitor Q values
            q_stats = {
                'q_max': q_values_all.max().item(),
                'q_min': q_values_all.min().item(),
                'q_mean': q_values_all.mean().item(),
            }
            
            self.targert_net.eval()  # Ensure target net is in inference mode
            with torch.no_grad():
                # Compute Q values of next state using target net
                next_state_q_values = self.target_net(torch.from_numpy(next_states).to(device)) 
        
                # Find maximum Q-value at next state using target network (standard DQN)
                next_state_max_q_values = torch.max(next_state_q_values, dim=1)[0]

                # Compute the target Q-value.  Disregard the next Q-value if the game is over
                target_q_values = rewards + self.discount_factor * next_state_max_q_values * (~terminations)
                target_q_values = target_q_values.unsqueeze(1)

            ## PER Logic ##
            if use_per:
                # Anneal Alpha and Beta
                self.beta = min(1.0, self.beta + (1.0-self.beta_min)/TRAINING_STEPS)
                self.alpha = max(0.2, self.alpha - (self.alpha_max - 0.2)/TRAINING_STEPS)

                # Update TD-errors in replay buffer
                td_errors = (q_values - target_q_values).squeeze(1).detach().cpu().numpy()
                td_errors = np.clip(np.abs(td_errors), 1e-6, 1)  # clip to avoid divide by zero and huge error spikes
                self.memory.update_td_errors(mini_batch_indices, td_errors)

                # Importance sampling to remove the bias of oversampling high TD-error states
                sampling_probs = np.clip(self.memory.get_sampling_probs(mini_batch_indices, alpha = self.alpha), 1e-7, None)  # Avoid 0 probabilites for division in next line
                is_weights = (1.0 / (len(self.memory.valid_indices) * sampling_probs)) ** self.beta
                is_weights /= is_weights.max()  # normalize
                is_weights = torch.FloatTensor(is_weights).to(device)

                # Use IS-weighted squared TD-error as loss with PER
                loss = (is_weights * (q_values.squeeze(1) - target_q_values.squeeze(1)) ** 2).mean()
            
            else:
                # Compute the Huber Loss for standard replay buffer
                loss_fn = nn.SmoothL1Loss()
                loss = loss_fn(q_values, target_q_values)

        # Optimize the model, .step() both the optimizer and the scheduler!
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss.item(), q_stats

    def save_replay_buffer(self, run_name, frame, dir="./checkpoints"):
        path = os.path.join(dir, run_name + '_' + str(frame) +'_replay_buffer.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)
        
        real_buffer_size = len(self.memory)
        print(f"Replay buffer of {real_buffer_size} frames saved")

    def load_replay_buffer(self, run_name, frame, dir="./checkpoints"):
        path = os.path.join(dir, run_name + '_' + str(frame) +'_replay_buffer.pkl')
        with open(path, 'rb') as f:
            self.memory = pickle.load(f)
        
        real_buffer_size = len(self.memory)
        print(f"Replay buffer of {real_buffer_size} frames loaded")

    def save_checkpoint(self, metadata, run_name, episode, dir="./checkpoints"): 
        torch_checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'torch_scheduler': self.scheduler.state_dict(),
            'metadata': metadata
        }

        torch_checkpoint['metadata']['epsilon'] = self.epsilon #add epislon to metadata

        path = os.path.join(dir, run_name + '_' + str(episode) +'_checkpoint.pt')
        torch.save(torch_checkpoint, path)

        print(f"Checkpoint for episode {episode} saved")


    def load_checkpoint(self, run_name, episode, dir="./checkpoints"):
        path = os.path.join(dir, run_name + '_' + str(episode) +'_checkpoint.pt')
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['torch_scheduler'])
        metadata = checkpoint['metadata']
        self.epsilon = metadata['epsilon']

        print(f"Checkpoint for episode {episode} loaded")
        return metadata
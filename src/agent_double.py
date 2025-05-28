import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import DQN
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os
import pickle
from agent import Agent as Agent_base
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(Agent_base):
    def __init__(self, action_size, replay_buffer_path=None):
        super().__init__(action_size, replay_buffer_path)
        self.epsilon = 1.0
        self.epsilon_max = self.epsilon
        self.epsilon_min = EPSILON_MIN
        self.explore_step = EXPLORE_STEPS 
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.epsilon_decay_rate = 0.998


    def get_agent_type(self):
        return "Double DQN Agent"


    def train_policy_net(self):

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            # Clamp in case of overshoot
            self.epsilon = max(self.epsilon, self.epsilon_min)

        # Sample batch and unpack
        mini_batch = self.memory.improved_sample_mini_batch(batch_size = BATCH_SIZE)
        mini_batch = np.array(mini_batch, dtype=object).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :HISTORY_SIZE, :, :]) / 255.
        next_states = np.float32(history[:, 1:, :, :]) / 255.

        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(list(mini_batch[1]), dtype=torch.long, device=device)
        rewards = torch.tensor(list(mini_batch[2]), dtype=torch.float32, device=device)
        terminations = torch.tensor(list(mini_batch[3]), dtype=torch.bool, device=device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)

        # Q(s, a) from policy network
        self.policy_net.train()
        q_values_all = self.policy_net(states)
        q_values = torch.gather(q_values_all, dim=1, index=actions.unsqueeze(1))

        # Monitor Q stats
        q_stats = {
            'q_max': q_values_all.max().item(),
            'q_min': q_values_all.min().item(),
            'q_mean': q_values_all.mean().item(),
        }

        with torch.no_grad():
            # Double DQN: use policy net to select best next action
            next_q_vals_policy = self.policy_net(next_states_tensor)
            next_argmax_action_policy = torch.argmax(next_q_vals_policy, dim=1)

            # Use target net to evaluate that action
            next_q_vals_target = self.target_net(next_states_tensor)
            next_q_tgt_for_policy_actn = torch.gather(
                next_q_vals_target, dim=1, index=next_argmax_action_policy.unsqueeze(1)
            ).squeeze(1)

            # Compute target Q-values (zero out future rewards if terminal)
            target_q_values = rewards + self.discount_factor * next_q_tgt_for_policy_actn * (~terminations)
            target_q_values = target_q_values.unsqueeze(1)


        # Compute loss (Huber loss)
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), q_stats

    


     
        
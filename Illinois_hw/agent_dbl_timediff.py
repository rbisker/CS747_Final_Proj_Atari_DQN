import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import DQN_DualBranch
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os
import pickle
from agent import Agent as Agent_base
import math
import matplotlib.pyplot as plt

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

        #swtich model to time diff version
        self.policy_net = DQN_DualBranch(action_size).to(device)
        self.target_net = DQN_DualBranch(action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        


    def get_agent_type(self):
        return "Double DQN Agent w time diff"
    
    
    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state: np.ndarray) -> int:
        
        # Based on epsilon, randomly decide whether to choose a random action or to choose the best action
        if np.random.rand() <= self.epsilon:
            a = np.random.choice(self.action_size)
        else:
            # construct input with timediff frames
            state_w_timediff = np.zeros((5, 84, 84), dtype=np.float32)
            state_w_timediff[:4] = state
            diff = (np.abs(state[3] - state[2]) * 40.0).clip(0.0, 1.0)
            if random.random() < 0.1:
                print(f"[DEBUG] Diff mean: {diff.mean():.5f}, std: {diff.std():.5f}, max: {diff.max():.2f}, min: {diff.min():.2f}")
            state_w_timediff[4] = diff
            # state_w_timediff[5] = (state[2] - state[1]) / 255.0
            # state_w_timediff[6] = (state[3] - state[2]) / 255.0
            state_tensor = torch.from_numpy(state_w_timediff).unsqueeze(0).to(device)
            
            # Choose the best action
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            a = torch.argmax(q_values, dim=1).item()
        return int(a)


    def train_policy_net(self):
        batch_size = BATCH_SIZE
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min) # Clamp in case of overshoot

        # Sample and unpack
        mini_batch = self.memory.improved_sample_mini_batch(batch_size = BATCH_SIZE)
        mini_batch = np.array(mini_batch, dtype=object).transpose()

        raw_histories = np.stack(mini_batch[0], axis=0)  # shape: (batch_size, 5, 84, 84)

        # # Construct 7-channel states and next_states
        # batch_size = raw_histories.shape[0]
        # states = np.zeros((batch_size, 7, 84, 84), dtype=np.float32)
        # next_states = np.zeros((batch_size, 7, 84, 84), dtype=np.float32)

        # for i in range(batch_size):
        #     h = raw_histories[i]
        #     s = h[:4].astype(np.float32) / 255.0
        #     s_diff = [
        #         s[1] - s[0],
        #         s[2] - s[1],
        #         s[3] - s[2]
        #     ]
        #     states[i, :4] = s
        #     states[i, 4] = s_diff[0]
        #     states[i, 5] = s_diff[1]
        #     states[i, 6] = s_diff[2]

        #     ns = h[1:].astype(np.float32) / 255.0
        #     ns_diff = [
        #         ns[1] - ns[0],
        #         ns[2] - ns[1],
        #         ns[3] - ns[2]
        #     ]
        #     next_states[i, :4] = ns
        #     next_states[i, 4] = ns_diff[0]
        #     next_states[i, 5] = ns_diff[1]
        #     next_states[i, 6] = ns_diff[2]

        # Construct 5-channel states with one diff frame
        states = np.zeros((batch_size, 5, 84, 84), dtype=np.float32)
        next_states = np.zeros((batch_size, 5, 84, 84), dtype=np.float32)

        for i in range(batch_size):
            s = raw_histories[i][:4].astype(np.float32) / 255.0
            diff = np.abs(s[3] - s[2]) * 40.0
            diff = np.clip(diff, 0.0, 1.0)

            ## DEBUG ##
            # if i == 0 and random.random() < 0.1:
            #     plt.imshow(diff, cmap='gray')
            #     plt.title("Diff Frame (scaled x40 and clipped)")
            #     plt.colorbar()
            #     plt.show()
            #     print(f"[DEBUG] Diff stats — mean: {diff.mean():.4f}, std: {diff.std():.4f}, min: {diff.min():.4f}, max: {diff.max():.4f}")
            ##########

            states[i, :4] = s
            states[i, 4] = diff

            ns = raw_histories[i][1:].astype(np.float32) / 255.0
            next_diff = ((ns[3] - ns[2]) * 10.0).clip(-1.0, 1.0)
            next_states[i, :4] = ns
            next_states[i, 4] = next_diff

        # Convert everything to torch
        states = torch.tensor(states, dtype=torch.float32, device=device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)
        actions = torch.tensor(list(mini_batch[1]), dtype=torch.long, device=device)
        rewards = torch.tensor(list(mini_batch[2]), dtype=torch.float32, device=device)
        terminations = torch.tensor(list(mini_batch[3]), dtype=torch.bool, device=device)

        # print(f"[DEBUG] Reward sample — mean: {rewards.mean().item():.4f}, max: {rewards.max().item():.1f}, nonzero: {(rewards != 0).sum().item()}")
        # print(f"[DEBUG] Diff stats — mean: {states[:, 4:].mean().item():.4f}, std: {states[:, 4:].std().item():.4f}")
        # Compute Q(s, a)
        self.policy_net.train()
        q_values_all = self.policy_net(states)
        q_values = torch.gather(q_values_all, dim=1, index=actions.unsqueeze(1))

        q_stats = {
            'q_max': q_values_all.max().item(),
            'q_min': q_values_all.min().item(),
            'q_mean': q_values_all.mean().item(),
        }

        with torch.no_grad():
            # Double DQN logic
            next_q_vals_policy = self.policy_net(next_states_tensor)
            next_argmax_action_policy = torch.argmax(next_q_vals_policy, dim=1)

            next_q_vals_target = self.target_net(next_states_tensor)
            next_q_tgt_for_policy_actn = torch.gather(
                next_q_vals_target, dim=1, index=next_argmax_action_policy.unsqueeze(1)
            ).squeeze(1)

            target_q_values = rewards + self.discount_factor * next_q_tgt_for_policy_actn * (~terminations)
            target_q_values = target_q_values.unsqueeze(1)

        # Compute loss and step
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        

        # if random.random() < 0.05:
        # # === Debug: Gradient flow ===
        #     print("[DEBUG] Gradients:")
        #     for name, param in self.policy_net.named_parameters():
        #         if param.grad is not None:
        #             print(f"  {name:25s} grad mean: {param.grad.abs().mean():.6f}")

        #     # === Debug: Target Q-values ===
        #     print(f"[DEBUG] Target Qs — mean: {target_q_values.mean().item():.4f}, std: {target_q_values.std().item():.4f}")

        #     # === Debug: Learning rate ===
        #     print(f"[DEBUG] Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        #     # === Debug: Q-values for current states (optional) ===
        #     print(f"[DEBUG] Predicted Qs — mean: {q_values_all.mean().item():.4f}, max: {q_values_all.max().item():.4f}, min: {q_values_all.min().item():.4f}")

        
        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), q_stats

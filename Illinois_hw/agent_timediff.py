import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory, ReplayMemory_PER
from model import DQN_DualBranch
from utils import *
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
        #swtich model to time diff version
        self.policy_net = DQN_DualBranch(action_size).to(device)
        self.target_net = DQN_DualBranch(action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.diff_scaling_factor = 1.5
        print("Setting up PER Replay Buffer")
        self.memory = ReplayMemory_PER()

    def get_agent_type(self):
        return "Standard DQN Agent w time diffs"
    
    def get_action(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        # state is already normalized (shape: [4, 84, 84])
        full = state  # no need to divide again

        # # Compute 2 difference frames (scaled and clipped)
        diffs = np.stack([
            full[1] - full[0],
            full[3] - full[2]
        ])

        # Normalize diffs
        diff_mean = np.mean(diffs)
        diff_std = np.std(diffs) + 1e-8 # avoid division by zero
        diffs = self.diff_scaling_factor * (diffs - diff_mean) / diff_std

        # Combine full frames and flow into 6-channel input
        state_6ch = np.concatenate([full, diffs], axis=0)  # shape: (6, 84, 84)
        state_tensor = torch.from_numpy(state_6ch).unsqueeze(0).to(device)  # add batch dim

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())
    
        
    def train_policy_net(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        # Sample and unpack
        if type(self.memory) == ReplayMemory:
            mini_batch = self.memory.improved_sample_mini_batch(batch_size=BATCH_SIZE)
        elif type(self.memory) == ReplayMemory_PER:
            mini_batch, mini_batch_indices = self.memory.improved_sample_mini_batch(batch_size=BATCH_SIZE)
        else:
            raise NotImplementedError("Invalid replay buffer type")

        mini_batch = np.array(mini_batch, dtype=object).transpose()

        raw_histories = np.stack(mini_batch[0], axis=0)  # (batch_size, 5, 84, 84)
        batch_size = raw_histories.shape[0]

        states = np.zeros((batch_size, 6, 84, 84), dtype=np.float32)
        next_states = np.zeros((batch_size, 6, 84, 84), dtype=np.float32)

        ## Construct optical flow input
        for i in range(batch_size):
            # === Current state ===
            s = raw_histories[i][:4].astype(np.float32) / 255.0  # (4, 84, 84)

            diffs = np.stack([
                s[1] - s[0],
                s[3] - s[2]
            ])

            diff_mean = np.mean(diffs)
            diff_std = np.std(diffs) + 1e-8 # avoid divide by zero
            diffs = self.diff_scaling_factor * (diffs - diff_mean) / diff_std

            
            ## DEBUG ##
            # if i == 0 and random.random() < 0.01:


            #     fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            #     axs[0].imshow(s[0], cmap='gray')
            #     axs[0].set_title('s[0]')
            #     axs[1].imshow(s[3], cmap='gray')
            #     axs[1].set_title('s[3]')
            #     plt.suptitle("Visual Check: Full Frames s[0] vs s[3]")
            #     plt.show()

            #     fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            #     axs[0].imshow(diffs[0], cmap='seismic', vmin=-1.0, vmax=1.0)
            #     axs[0].set_title('Diff: s[1] - s[0]')
            #     axs[1].imshow(diffs[1], cmap='seismic', vmin=-1.0, vmax=1.0)
            #     axs[1].set_title('Diff: s[3] - s[2]')
            #     plt.suptitle("Signed Frame Differences")
            #     plt.show()
            
            states[i, :4] = s
            states[i, 4:] = diffs

            # if i == 0 and random.random() < 0.1:
            #     print(f"[DEBUG] Diff mean: {diffs.mean():.5f}, std: {diffs.std():.5f}, max: {diffs.max():.2f}, min: {diffs.min():.2f}")
            #     print(f"[DEBUG] Full mean: {s.mean():.5f}, std: {s.std():.5f}, max: {s.max():.2f}, min: {s.min():.2f}")

            ns = raw_histories[i][1:].astype(np.float32) / 255.0
            next_diffs = np.stack([
                ns[1] - ns[0],
                ns[3] - ns[2],
            ])

            next_diff_mean = np.mean(next_diffs)
            next_diff_std = np.std(next_diffs) + 1e-8 # avoid divide by zero
            next_diffs = self.diff_scaling_factor * (next_diffs - next_diff_mean) / next_diff_std

            next_states[i, :4] = ns
            next_states[i, 4:] = next_diffs
            ## DEBUG ##
            # if i == 0 and random.random() < 0.02:
            #     print(f"[DIFF DEBUG] diff1 mean: {diffs[0].mean():.4f}, std: {diffs[0].std():.4f} | diff2 mean: {diffs[1].mean():.4f}, std: {diffs[1].std():.4f}")
            #     print(f"[STATE DEBUG] state shape: {states[i].shape}, full mean: {np.mean(states[i][:4]):.4f}, diff mean: {np.mean(states[i][4:]):.4f}")

        # Convert to torch tensors
        states = torch.tensor(states, dtype=torch.float32, device=device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)
        actions = torch.tensor(list(mini_batch[1]), dtype=torch.long, device=device)
        rewards = torch.tensor(list(mini_batch[2]), dtype=torch.float32, device=device)
        terminations = torch.tensor(list(mini_batch[3]), dtype=torch.bool, device=device)

        ## DEBUG ##
        if random.random() < 0.001:
            with torch.no_grad():
                sample = states[:1]  # Grab just the first state from the minibatch
                out = self.policy_net(sample)  # Run a forward pass
                print("[DEBUG] Q-values:", out.cpu().numpy(), "mean:", out.mean().item(), "std:", out.std().item(), "max:", out.max().item())

        # Forward pass
        self.policy_net.train()
        q_values_all = self.policy_net(states)
        
        # ## DEBUG ##
        # if random.random() < 0.05:
        #     for name, param in self.policy_net.named_parameters():
        #         if 'fc' in name or 'head' in name:
        #             print(f"[WEIGHT INIT] {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")
        
        q_values = torch.gather(q_values_all, dim=1, index=actions.unsqueeze(1))

        #Monitor Q values
        q_stats = {
            'q_max': q_values_all.max().item(),
            'q_min': q_values_all.min().item(),
            'q_mean': q_values_all.mean().item(),
            'q_std': q_values_all.std().item()
        }

        ## DEBUG ##
        # if random.random() < 0.05:
        #     print(f"[Q-VALUE STATS] mean: {q_stats['q_mean']:.3f} | std: {q_stats['q_std']:.3f} | min: {q_stats['q_min']:.3f} | max: {q_stats['q_max']:.3f}")
        
        with torch.no_grad():
            # Compute Q values of next state using target net
            next_state_q_values = self.target_net(next_states_tensor)
       
            # Find maximum Q-value at next state using target network (standard DQN)
            next_state_max_q_values = torch.max(next_state_q_values, dim=1)[0]

            # Compute the target Q-value.  Disregard the next Q-value if the game is over
            target_q_values = rewards + self.discount_factor * next_state_max_q_values * (~terminations)
            target_q_values = target_q_values.unsqueeze(1)

            #If Using PER, update memory td_errors
            if type(self.memory) == ReplayMemory_PER:
                td_errors = np.abs((q_values - target_q_values).squeeze(1).detach().cpu().numpy())
                self.memory.update_td_errors(mini_batch_indices, td_errors)
        
        # Compute the Huber Loss
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(q_values, target_q_values)

        # Optimize the model, .step() both the optimizer and the scheduler!
        self.optimizer.zero_grad()
        loss.backward()

        # ## DEBUG ##
        # if random.random() < 0.05:
        #     print("[GRADIENT CHECK]")
        #     for name, param in self.policy_net.named_parameters():
        #         if param.grad is not None and ('fc' in name or 'head' in name):
        #             print(f"{name} grad std: {param.grad.std().item():.6f}")

        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), q_stats
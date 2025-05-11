from config import *
from collections import deque
import numpy as np
import random
import torch
from concurrent.futures import ThreadPoolExecutor


class ReplayMemory(object):
    def __init__(self):
        self.memory = deque(maxlen=Memory_capacity)
    
    def push(self, history, action, reward, done, td_error = 1.0):
        self.memory.append((history, action, reward, done, td_error))
    
    '''
    Improved to avoid samping invalid transitions (where any of the first HISTORY_SIZE frames end in a life loss)
    '''
    def improved_sample_mini_batch(self, batch_size = BATCH_SIZE, HISTORY_SIZE=4, max_tries_per_sample=50):
        mini_batch = []
        used_indices = set()
        sample_range = len(self.memory) - (HISTORY_SIZE + 1)

        if sample_range < batch_size:
            raise ValueError("Not enough samples in replay buffer to form a valid mini-batch.")

        tries = 0
        while len(mini_batch) < batch_size and tries < max_tries_per_sample * batch_size:
            tries += 1
            i = random.randint(0, sample_range - 1)
            
            # Reject this index if previously used OR if any of the first HISTORY_SIZE frames after idx ends in a life loss
            if any(self.memory[i + j][3] for j in range(HISTORY_SIZE)) or i in used_indices:     
                # print("[DEBUG] index rejected")
                continue  # Try another index

            #If valid, build the the sample and add to mini_batch
            used_indices.add(i)
            
            sample = [self.memory[i + j] for j in range(HISTORY_SIZE + 1)]
            sample = np.array(sample, dtype=object)

            states = np.stack(sample[:, 0], axis=0)
            actions = int(sample[HISTORY_SIZE - 1, 1])
            rewards = float(sample[HISTORY_SIZE - 1, 2])
            terminations = bool(sample[HISTORY_SIZE - 1, 3])

            mini_batch.append((states, actions, rewards, terminations))
            
        if len(mini_batch) < batch_size:
            print(f"[WARNING] Only sampled {len(mini_batch)} valid transitions out of requested {batch_size}. Consider increasing memory or improving episode termination tracking.")

        return mini_batch

    def __len__(self):
        return len(self.memory)



'''
Switched replay buffer from deque to a cirular buffer to allow for effecient tracking of valid states and td-errors in 
a manner that was impossible to do with deques without requiring an O(n) scan through buffer every training step to search for valid
states
'''
class CircularReplayMemoryPER:
    def __init__(self, capacity, history_size):
        self.capacity = capacity
        self.history_size = history_size
        self.memory = [None] * capacity
        self.td_errors = [] # TD-errors for valid transitions
        self.valid_flags = [False] * capacity # True if the next HISTORY_SIZE frames forward are not terminal
        self.valid_indices = []  # Only true for frames that are the start of a valid state
        self.position = 0
        self.size = 0


        #Priority Caching
        self._priority_cache_dirty = True
        self._cached_probs = []  # sampling probabilities fo valid indices, should be length of len(self.valid_indices)

    def push(self, frame, action, reward, done, mean_recent_td_error=1.0):
        i = self.position
        # Convert frame to float16 if it's a NumPy array and not already float16
        if isinstance(frame, np.ndarray) and frame.dtype != np.float16:  
            frame = frame.astype(np.float16)
        self.memory[i] = (frame, action, reward, done)
        self.valid_flags[i] = False  # Placeholder, updated retroactively
        

        # Retroactively validate a sample at index (i - history_size)
        valid_start = (i - self.history_size) % self.capacity
        if self.size >= self.history_size + 1:
            is_valid = True
            for j in range(self.history_size):
                check_idx = (valid_start + j) % self.capacity
                if self.memory[check_idx][3]:  # If `done` is True
                    is_valid = False
                    break
            self.valid_flags[valid_start] = is_valid
            if is_valid:
                self.valid_indices.append(valid_start)
                
                # use policy to set TD-error of newly validated index [ADDS SIGNIFICANT TRAINING TIME]
                # state = np.stack([(self.memory[(valid_start + j) % self.capacity][0])/255.0 for j in range(self.history_size)], axis=0)
                # next_state = np.stack([(self.memory[(valid_start + 1 + j) % self.capacity][0])/255.0 for j in range(self.history_size)], axis=0)
                # action = self.memory[(valid_start + self.history_size -1) % self.capacity][1]
                # reward = self.memory[(valid_start + self.history_size -1) % self.capacity][2]
                # done = self.memory[(valid_start + self.history_size -1) % self.capacity][3]
                # self.td_errors.append(self.get_td_error(agent, state, action, reward, done, next_state))  # set TD-error for valid index to policy estimated TD-error

                # set td-error to mean of recent TD-errors [FASTER!]
                self.td_errors.append(mean_recent_td_error)
                

        

        # Handle validity buffer after memory is full and has looped
        if self.size == self.capacity:
            overwritten_index = self.position
            if overwritten_index in self.valid_indices:
                idx = self.valid_indices.index(overwritten_index)
                del self.valid_indices[idx] # Remove the overwritten index from valid list
                del self.td_errors[idx]     # Remove associated TD-error
            self.valid_flags[overwritten_index] = False  # reset overwritten index valid flag to False

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._priority_cache_dirty = True

    """
    Improved version of sample_mini_batch that uses priority experience replay (PER) based on TD-errors 
    and avoids including sampling states that cross a lost life frame.
    :param batch_size: The size of the mini-batch to sample.
    :param HISTORY_SIZE: The number frames that make up a state passed to the network
    :return: A list of tuples containing the states, actions, rewards, and termination signals for each sample in the mini-batch.
    """
    def improved_sample_mini_batch(self, batch_size, per_alpha=PER_ALPHA):
        assert len(self.valid_indices) == len(self.td_errors) == sum(self.valid_flags), \
            "Before creating mimi-batch, # of valid indices, # of TD-errors and count of True valid flags must all be equal"
        if len(self.valid_indices) < batch_size:
            print("Length of valid indices:", len(self.valid_indices))
            raise ValueError("Not enough valid samples.")

        # get sampling probabilities weighted by TD-errors
        probs = self.get_sampling_probs(self.valid_indices, per_alpha)
        
        cdf = np.cumsum(probs)
        valid_indices_idxs = np.searchsorted(cdf, np.random.rand(batch_size), side='right')  # get indices of where a batch size of rands fall in the cdf
        valid_indices_idxs = np.clip(valid_indices_idxs, 0, len(self.valid_indices) - 1)  # for safety reasons, make sure we don't go out of bounds
        sample_valid_indices = [self.valid_indices[i] for i in valid_indices_idxs]  # get the actual valid_indices


        # mini_batch = []
        # for idx in sample_valid_indices:
        #     sample = [(self.memory[(idx + j) % self.capacity]) for j in range(self.history_size + 1)]
        #     sample = np.array(sample, dtype=object)

        #     states = np.stack(sample[:, 0], axis=0)
        #     actions = int(sample[self.history_size - 1, 1])
        #     rewards = float(sample[self.history_size - 1, 2])
        #     terminations = bool(sample[self.history_size - 1, 3])
        
        def build_sample(idx):
            sample = [(self.memory[(idx + j) % self.capacity]) for j in range(self.history_size + 1)]
            sample = np.array(sample, dtype=object)
            states = np.stack(sample[:, 0], axis=0)
            actions = int(sample[self.history_size - 1, 1])
            rewards = float(sample[self.history_size - 1, 2])
            terminations = bool(sample[self.history_size - 1, 3])
            return (states, actions, rewards, terminations)

        with ThreadPoolExecutor(max_workers=4) as executor:
            mini_batch = list(executor.map(build_sample, sample_valid_indices))


        ## DEBUG ###
        # if random.random() < 0.001:
            # # Print a few sampled TD errors
            # sampled_td_errors = [self.td_errors[i] for i in sample_indices[:5]]
            # print("[PER DEBUG] Sampled TD-errors (first 5):", sampled_td_errors)
        
            # Check validity
            # print("[PER DEBUG] Len of valid indices:", len(self.valid_indices), "  Len of valid flags:", sum(self.valid_flags))
        ################################################

        return mini_batch, valid_indices_idxs
    
    """
    Given a list of sampled indices, return their sampling probabilities
    under the current priority distribution (with exponent alpha).
    """
    def get_sampling_probs(self, valid_indices_idxs, alpha=PER_ALPHA):

        if self._priority_cache_dirty:
            # Get sampling probabilities for all valid indices
            all_priorities = np.power(self.td_errors, alpha)
            total_priority = all_priorities.sum()
            self._cached_probs = all_priorities / total_priority if total_priority > 0 else np.ones_like(all_priorities)/len(self.valid_indices)  
            assert len(self._cached_probs) == len(self.valid_indices), "On priority cache update, len(self._cached_probs) != len(self.valid_indices)"
            self._priority_cache_dirty = False

        # If all full valid indices were passed, return full list of sampling probs
        if np.array_equal(valid_indices_idxs, self.valid_indices):
            return self._cached_probs
        else: 
            # Extract sampling probs for just the passed indices
            sample_probs = np.array([self._cached_probs[i] for i in valid_indices_idxs], dtype=np.float32)
            return sample_probs


    def update_td_errors(self, valid_indices_idxs, new_errors):
        for i, err in zip(valid_indices_idxs, new_errors):
            self.td_errors[i] = err
        self._priority_cache_dirty = True

    def __len__(self):
        return self.size
    
    def log_td_error_distribution(self):
        print("[PER STATS] TD-error mean:", np.mean(self.td_errors),
            "std:", np.std(self.td_errors),
            "min:", np.min(self.td_errors),
            "max:", np.max(self.td_errors))
        
    def get_td_error(self, agent, state, action, reward, done, next_state):
        with torch.no_grad():
                    state_tensor = torch.from_numpy(state).unsqueeze(0).float().cuda()
                    next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).float().cuda()
                
                    # Q(s, a) under policy net
                    q_values = agent.policy_net(state_tensor)
                    q_sa = q_values[0, action]

                    # Compute target Q(s', a') under target net
                    next_q_values = agent.target_net(next_state_tensor)
                    next_q_max = torch.max(next_q_values, dim=1)[0]  # max over actions

                    target = reward + (0 if done else agent.discount_factor * next_q_max.item())

                    td_error = abs(q_sa.item() - target)
                    # td_error = np.clip(td_error, 1e-6, 1)  # clip to avoid huge error spikes

        return td_error




class ReplayMemoryLSTM(ReplayMemory):
    """
    This is a version of Replay Memory modified for LSTMs. 
    Replay memory generally stores (state, action, reward, next state).
    But LSTMs need sequential data. 
    So we store (state, action, reward, next state) for previous few states, constituting a trajectory.
    During training, the previous states will be used to generate the current state of LSTM. 
    Note that samples from previous episode might get included in the trajectory.
    Inspite of not being fully correct, this simple Replay Buffer performs well.
    """
    def __init__(self):
        super().__init__()

    def sample_mini_batch(self, frame):
        mini_batch = []
        if frame >= Memory_capacity:
            sample_range = Memory_capacity
        else:
            sample_range = frame

        sample_range -= (lstm_seq_length + 1)

        idx_sample = random.sample(range(sample_range - lstm_seq_length), batch_size)
        for i in idx_sample:
            sample = []
            for j in range(lstm_seq_length + 1):
                sample.append(self.memory[i + j])

            sample = np.array(sample, dtype=object)
            mini_batch.append((np.stack(sample[:, 0], axis=0), sample[lstm_seq_length - 1, 1], sample[lstm_seq_length - 1, 2], sample[lstm_seq_length - 1, 3]))

        return mini_batch

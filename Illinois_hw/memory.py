from config import *
from collections import deque
import numpy as np
import random


class ReplayMemory(object):
    def __init__(self):
        self.memory = deque(maxlen=Memory_capacity)
    
    def push(self, history, action, reward, done, td_error = 1.0):
        self.memory.append((history, action, reward, done, td_error))
    
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

class ReplayMemory_PER(object):
    def __init__(self):
        self.memory = deque(maxlen=Memory_capacity)
        self.valid_flags = deque(maxlen=Memory_capacity) # Parallel flag queue
       
    
    def push(self, history, action, reward, done, td_error = 1.0):  
        self.memory.append((history, action, reward, done, td_error))
        self.valid_flags.append(False)  # placeholder, updated retroactively
                
        # Look back HISTORE_SIZE frames and check if the sample is valid by scanning HISTORY_SIZE forward steps
        if len(self.memory) >= HISTORY_SIZE + 1:
            base_idx = len(self.memory) - (HISTORY_SIZE + 1)
            if base_idx >= 0 and not any(self.memory[base_idx + j][3] for j in range(HISTORY_SIZE)):
                self.valid_flags[-(HISTORY_SIZE + 1)] = True
                

    """
    Improved version of sample_mini_batch that uses priority sampling based on TD-errors and avoids including samples that cross a lost life.
    :param batch_size: The size of the mini-batch to sample.
    :param HISTORY_SIZE: The number of frames in the history of the environment.
    :return: A list of tuples containing the state history, action, reward, and termination signal for each sample in the mini-batch.
    """
    def improved_sample_mini_batch(self, batch_size = BATCH_SIZE, history_size = HISTORY_SIZE):
        assert len(self.memory) == len(self.valid_flags), "Repaly buffer and valid flags must have the same length."
        mini_batch = []
        valid_indices = [i for i, val in enumerate(self.valid_flags) if val]
        
        if len(valid_indices) < batch_size:
            raise ValueError("Not enough valid samples in replay buffer to form a valid mini-batch.")
        
        
        probs = [self.memory[i][4] for i in valid_indices]
        idx_sample = random.choices(valid_indices, weights=probs, k=batch_size)

        for i in idx_sample:
            sample = [self.memory[i + j] for j in range(history_size + 1)]
            sample = np.array(sample, dtype=object)

            states = np.stack(sample[:, 0], axis=0)
            actions = int(sample[history_size - 1, 1])
            rewards = float(sample[history_size - 1, 2])
            terminations = bool(sample[history_size - 1, 3])

            mini_batch.append((states, actions, rewards, terminations))


        ## DEBUG ###
        if random.random() < 0.002:
            # Print a few sampled TD errors
            sampled_td_errors = [self.memory[i][4] for i in idx_sample[:5]]
            print("[PER DEBUG] Sampled TD-errors (first 5):", sampled_td_errors)

            # Ensure validity: no `done` flag in history frames
            for i in idx_sample[:5]:
                termination_flags = [self.memory[i + j][3] for j in range(history_size)]
                if any(termination_flags):
                    print(f"[PER WARNING] Invalid sample at index {i}: {termination_flags}")

        ################################################
            
        return mini_batch, idx_sample

 
    def update_td_errors(self, indices, new_td_errors):
        for idx, error in zip(indices, new_td_errors):
            # Convert the tuple to list to modify
            sample = list(self.memory[idx])
            sample[4] = error
            self.memory[idx] = tuple(sample)

    def __len__(self):
        return len(self.memory)
    
    def log_td_error_distribution(self):
        td_errors = [item[4] for item in self.memory]
        if td_errors:
            print("[PER STATS] TD-error mean:", np.mean(td_errors),
                "std:", np.std(td_errors),
                "min:", np.min(td_errors),
                "max:", np.max(td_errors))


class CircularReplayMemoryPER:
    def __init__(self, capacity, history_size):
        self.capacity = capacity
        self.history_size = history_size
        self.memory = [None] * capacity
        self.td_errors = np.ones(capacity, dtype=np.float32)
        self.valid_flags = [False] * capacity
        self.valid_indices = []  # Only valid sampling start indices
        self.position = 0
        self.size = 0

    def push(self, history, action, reward, done, td_error=1.0):
        i = self.position
        self.memory[i] = (history, action, reward, done)
        self.td_errors[i] = td_error
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

        # Handle validity buffer after memory is full and has looped
        if self.size == self.capacity:
            overwritten_index = self.position
            if self.valid_flags[overwritten_index]:
                try:
                    self.valid_indices.remove(overwritten_index) # Remove the overwritten index from valid list
                except ValueError:
                    pass  # Already removed
                self.valid_flags[overwritten_index] = False  # reset overwritten index valid flag to False

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def improved_sample_mini_batch(self, batch_size):
        # assert sum(self.valid_flags) == len(self.valid_indices), "Valid indices must match number of true valid flags"
        if len(self.valid_indices) < batch_size:
            raise ValueError("Not enough valid samples.")

        # === PER sampling via cumulative priority ===
        priors = np.array([self.td_errors[i] for i in self.valid_indices], dtype=np.float32)
        priors_sum = priors.sum()
        if priors_sum == 0:
            probs = np.ones_like(priors) / len(priors)
        else:
            probs = priors / priors_sum

        cum_probs = np.cumsum(probs)
        rand_vals = np.random.rand(batch_size) * cum_probs[-1]
        sample_ids = np.searchsorted(cum_probs, rand_vals)
        sample_indices = [self.valid_indices[i] for i in sample_ids]

        mini_batch = []
        for idx in sample_indices:
            sample = [(self.memory[(idx + j) % self.capacity]) for j in range(self.history_size + 1)]
            sample = np.array(sample, dtype=object)

            states = np.stack(sample[:, 0], axis=0)
            actions = int(sample[self.history_size - 1, 1])
            rewards = float(sample[self.history_size - 1, 2])
            terminations = bool(sample[self.history_size - 1, 3])
            mini_batch.append((states, actions, rewards, terminations))

        ## DEBUG ###
        if random.random() < 0.001:
            # # Print a few sampled TD errors
            # sampled_td_errors = [self.td_errors[i] for i in sample_indices[:5]]
            # print("[PER DEBUG] Sampled TD-errors (first 5):", sampled_td_errors)
        
            # Check validity
            print("[PER DEBUG] Len of valid indices:", len(self.valid_indices), "  Len of valid flags:", sum(self.valid_flags))
        ################################################

        return mini_batch, sample_indices

    def update_td_errors(self, indices, new_errors):
        for i, err in zip(indices, new_errors):
            self.td_errors[i] = err

    def __len__(self):
        return self.size
    
    def log_td_error_distribution(self):
        print("[PER STATS] TD-error mean:", np.mean(self.td_errors),
            "std:", np.std(self.td_errors),
            "min:", np.min(self.td_errors),
            "max:", np.max(self.td_errors))



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

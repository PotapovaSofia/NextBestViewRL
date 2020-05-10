import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from collections import deque


class DiskReplayBuffer:
    def __init__(
            self,
            capacity,
            observation_shape,
            observation_dtype,
            num_actions,
            overwrite=False,
            location='.',
            name='replay_buffer'):
        self._observation_shape = observation_shape
        self._observation_dtype = observation_dtype
        self._num_actions = num_actions
        self.capacity = capacity

        file_path = os.path.join(location, name)
        
        mode= 'w+'
        if os.path.exists(f'{file_path}.obs') and not overwrite:
            mode = 'r+'

        self._states = np.memmap(f'{file_path}.obs', dtype=observation_dtype, mode=mode,
                                       shape=(capacity,) + observation_shape)
        # TODO optimize
        self._next_states = np.memmap(f'{file_path}.nextobs', dtype=observation_dtype, mode=mode,
                                      shape=(capacity,) + observation_shape)
        self._actions = np.memmap(f'{file_path}.actions', dtype=np.int32, mode=mode,
                                  shape=(capacity, 1))
        self._rewards = np.memmap(f'{file_path}.rewards', dtype=np.float32, mode=mode,
                                  shape=(capacity, 1))
        self._done = np.memmap(f'{file_path}.done', dtype=np.uint8, mode=mode,
                               shape=(capacity, 1))
        self._mask = np.memmap(f'{file_path}.mask', dtype=bool, mode=mode,
                               shape=(capacity, self._num_actions))

        self._top = 0
        self._size = 0

    def __len__(self):
        return self._size

    def push(self, state, action, reward, next_state, done, mask):
        self._states[self._top]      = state
        self._actions[self._top]     = action
        self._rewards[self._top]     = reward
        self._next_states[self._top] = next_state
        self._done[self._top]        = done
        self._mask[self._top]        = mask

        self._top = (self._top + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, batch_size=1):
        indices = np.random.choice(self._size, batch_size)
        
        state      = self._states[indices]
        action     = self._actions[indices]
        reward     = self._rewards[indices]
        next_state = self._next_states[indices]
        done       = self._done[indices]
        mask       = self._mask[indices]

        state      = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action     = torch.tensor(np.concatenate(action), dtype=torch.long)
        reward     = torch.tensor(np.concatenate(reward), dtype=torch.float)
        done       = torch.tensor(np.concatenate(done), dtype=torch.float)
        mask       = torch.tensor(mask, dtype=torch.float)
            
        return state, action, reward, next_state, done, mask    


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, state, action, reward, next_state, done, mask):

        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        mask       = np.expand_dims(mask, 0)

        self.buffer.append((state, action, reward, next_state, done, mask))

    def sample(self, batch_size=1):
        state, action, reward, next_state, done, mask = zip(*random.sample(self.buffer, batch_size))

        """
        state      = torch.cat(state)
        action     = torch.cat(action)
        reward     = torch.cat(reward)
        next_state = torch.cat(next_state)
        done       = torch.cat(done)
        mask       = torch.cat(mask)
        """

        state      = torch.tensor(np.concatenate(state), dtype=torch.float)
        action     = torch.tensor(action, dtype=torch.long)
        reward     = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(np.concatenate(next_state), dtype=torch.float)
        done       = torch.tensor(done, dtype=torch.float)
        mask       = torch.tensor(np.concatenate(mask), dtype=torch.float)

        return state, action, reward, next_state, done, mask
    
    def __len__(self):
        return len(self.buffer)



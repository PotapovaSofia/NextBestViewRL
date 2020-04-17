import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from collections import deque

USE_CUDA = torch.cuda.is_available()

def variable_fun(device):
    return lambda *args, **kwargs: autograd.Variable(*args, **kwargs).to(device) \
        if USE_CUDA else autograd.Variable(*args, **kwargs)


cnn_device = torch.cuda.current_device()
Variable = variable_fun(cnn_device)

class BufferDevicesBalancer:
    def __init__(self, config={"cuda:0": 200, "cuda:0": 200, "cpu": 5000}):
        self.buffers = []

        for name, capacity in config.items():
            if name.split(":")[0] == "cuda":
                replay_buffer = ReplayBuffer(capacity, variable_fun(name), use_gpu=True)
            else:
                replay_buffer = ReplayBuffer(capacity, use_gpu=False)
            self.buffers.append(replay_buffer)

        lens = np.asarray([buffer.capacity for buffer in self.buffers])
        self.probs = lens / np.sum(lens)
        
    def push(self, state, action, reward, next_state, done): #*args, **kwargs
        buffer = self._get_buffer()
        buffer.push(state, action, reward, next_state, done)
    
    def sample(self, batch_size):
        buffer = self._get_buffer()
        while True:
            try:
                return buffer.sample(batch_size)
            except:
                continue
    
    def _get_buffer(self):
        return np.random.choice(self.buffers, p=self.probs)
            
    def __len__(self):
        return sum([len(buffer) for buffer in self.buffers])


class DiskReplayBuffer:
    def __init__(
            self,
            capacity,
            observation_shape,
            observation_dtype,
            overwrite=False,
            location='.',
            name='replay_buffer'):
        self._observation_shape = observation_shape
        self._observation_dtype = observation_dtype
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

        self._top = 0
        self._size = 0

    def __len__(self):
        return self._size

    def push(self, state, action, reward, next_state, done):
        self._states[self._top] = state
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._next_states[self._top] = next_state
        self._done[self._top] = done

        self._top = (self._top + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, batch_size=1):
        indices = np.random.choice(self._size, batch_size)
        
        state, action, reward, next_state, done = [], [], [], [], []
        for index in indices:
            state.append(self._states[index])
            action.append(self._actions[index])
            reward.append(self._rewards[index])
            next_state.append(self._next_states[index])
            done.append(self._done[index])
            
        state      = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)),
                              volatile=True)
        action     = Variable(torch.LongTensor(np.concatenate(action)))
        reward     = Variable(torch.FloatTensor(np.concatenate(reward)))
        done       = Variable(torch.FloatTensor(np.concatenate(done)))
            
        return state, action, reward, next_state, done    


class ReplayBuffer:
    def __init__(self, capacity, Variable=None, use_gpu=False):
        self.buffer = deque(maxlen=capacity)
        self.use_gpu = use_gpu
        self.capacity = capacity
        
        self.Variable = Variable
        if self.Variable is None:
            self.Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
                if self.use_gpu else lambda *args, **kwargs: autograd.Variable(*args, **kwargs)
            
                
    def push(self, state, action, reward, next_state, done):

        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        if self.use_gpu:
            state      = self.Variable(torch.FloatTensor(np.float32(state)))
            next_state = self.Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
            action     = self.Variable(torch.LongTensor([action]))
            reward     = self.Variable(torch.FloatTensor([reward]))
            done       = self.Variable(torch.FloatTensor([done]))

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=1):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        if self.use_gpu:
            # QUESTION about devices
            state      = torch.cat(state).to(cnn_device)
            action     = torch.cat(action).to(cnn_device)
            reward     = torch.cat(reward).to(cnn_device)
            next_state = torch.cat(next_state).to(cnn_device)
            done       = torch.cat(done).to(cnn_device)
        else:
            state      = Variable(torch.FloatTensor(np.float32(np.concatenate(state))))
            next_state = Variable(torch.FloatTensor(np.float32(np.concatenate(next_state))),
                                  volatile=True)
            action     = Variable(torch.LongTensor(action))
            reward     = Variable(torch.FloatTensor(reward))
            done       = Variable(torch.FloatTensor(done))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)



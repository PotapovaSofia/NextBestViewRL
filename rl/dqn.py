import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from rl.utils import variable_fun

USE_CUDA = torch.cuda.is_available()
Variable = variable_fun(torch.cuda.current_device())
Variable = variable_fun(1)


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.num_actions)
        return action
    
    
    
class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions, gamma=0.99, learning_rate=0.001):
        super().__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].item()
            print("Action: ", action)
        else:
            action = random.randrange(self.num_actions)
            print("Action: ", action, "(random)")
        return action
    
    def compute_td_loss(self, state, action, reward, next_state, done):
        q_values      = self.forward(state)
        next_q_values = self.forward(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = torch.mean((q_value - expected_q_value.detach()) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


class CnnDQNA(nn.Module):
    def __init__(self, input_shape, num_actions, gamma=0.99, learning_rate=0.001):
        super().__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        
        self.image_vector_size = 512
        
        self.cnn_features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc_features = nn.Sequential(
            nn.Linear(self.feature_size(), self.image_vector_size),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.image_vector_size + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, state, action):
        state = self.cnn_features(state)
        state = state.view(state.size(0), -1)
        state = self.fc_features(state)
        
        action = action.view(action.shape[0], 1)
        
        x = torch.cat((state, action), 1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.cnn_features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            q_values = []
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            for action in range(self.num_actions):
                action  = Variable(torch.FloatTensor([action]).unsqueeze(0), volatile=True)
                q_value = self.forward(state, action)
                q_values.append(q_value.item())
            action  = np.argmax(q_values)
            print("Action: ", action)
        else:
            action = random.randrange(self.num_actions)
            print("Action: ", action, "(random)")
        return action

    def compute_td_loss(self, state, action, reward, next_state, done):
        q_value = self.forward(state, action).squeeze(1)

        next_q_values = []
        for batch_idx in range(state.shape[0]):
            next_q_value = -np.inf
            for next_action in range(self.num_actions):
                next_action = Variable(torch.FloatTensor([next_action])).unsqueeze(0)
                pred = self.forward(next_state[batch_idx].unsqueeze(0), next_action).item()
                next_q_value = max(next_q_value, pred)
            next_q_values.append(next_q_value)

        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = torch.mean((q_value - expected_q_value) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


class VoxelDQN(nn.Module):
    def __init__(self, input_shape, num_actions, gamma=0.99, learning_rate=0.001):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].item()
            print("Action: ", action)
        else:
            action = random.randrange(self.num_actions)
            print("Action: ", action, "(random)")
        return action

    def compute_td_loss(self, state, action, reward, next_state, done):
        q_values      = self.forward(state)
        next_q_values = self.forward(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = torch.mean((q_value - expected_q_value.detach()) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


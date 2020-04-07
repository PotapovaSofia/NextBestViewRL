import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
    if USE_CUDA else autograd.Variable(*args, **kwargs)


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
            action = random.randrange(env.action_space.n)
        return action
    
    
    
class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        
        self.input_shape = input_shape
        # TODO action input
        self.num_actions = num_actions
        
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
        
        # concat with action_num
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
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
            action = random.randrange(env.action_space.n)
            print("Action: ", action, "(random)")
        return action
    
    @staticmethod
    def compute_td_loss(batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        q_values      = model(state)
        next_q_values = model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = torch.mean((q_value - expected_q_value.detach()) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss


class CnnDQNA(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
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
            for action in range(self.num_actions):
                state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
                action  = Variable(torch.FloatTensor([action]).unsqueeze(0), volatile=True)
                q_value = self.forward(state, action)
                q_values.append(q_value.item())
            best_action  = np.argmax(q_values)
            print("Action: ", action)
        else:
            action = random.randrange(env.action_space.n)
            print("Action: ", action, "(random)")
        return action

    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        q_value = model(state, action).squeeze(1)

        next_q_values = []
        for batch_idx in range(batch_size):
            next_q_value = -np.inf
            for action in range(self.num_actions):
                next_action = Variable(torch.FloatTensor([next_action])).unsqueeze(0)
                pred = model(next_state[batch_idx].unsqueeze(0), next_action).item()
                next_q_value = max(next_q_value, pred)
            next_q_values.append(next_q_value)

        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = torch.mean((q_value - expected_q_value) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

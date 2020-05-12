import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rl.dqn import DQN, CnnDQN, CnnDQNA, VoxelDQN


class ContDQNAgent:
    def __init__(self, observation_shape, num_actions, gamma=0.99, learning_rate=0.001):
        super().__init__()
        
        self.num_actions = num_actions
        self.gamma = gamma
        
        self.model = CnnDQNA(observation_shape, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

        
    def act(self, state, epsilon=0.05):
        if random.random() > epsilon:
            q_values = []
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            if self.use_cuda:
                state = state.cuda()
            for action in range(self.num_actions):
                action = torch.tensor([action], dtype=torch.float).unsqueeze(0)
                q_value = self.model(state, action)
                q_values.append(q_value.item())
            action  = np.argmax(q_values)
            # print("Action: ", action)
        else:
            action = random.randrange(self.num_actions)
            # print("Action: ", action, "(random)")
        return action

    def compute_td_loss(self, state, action, reward, next_state, done):
        if self.use_cuda:
            state = state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            next_state = next_state.cuda()
            done = done.cuda()
            mask = mask.cuda()


        q_value = self.model(state, action).squeeze(1)

        next_q_values = []
        for batch_idx in range(state.shape[0]):
            next_q_value = -np.inf
            for next_action in range(self.num_actions):
                next_action = torch.tensor([next_action], dtype=torch.float).unsqueeze(0)
                pred = self.model(next_state[batch_idx].unsqueeze(0), next_action).item()
                next_q_value = max(next_q_value, pred)
            next_q_values.append(next_q_value)

        expected_q_value = reward + self.gamma * next_q_values * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class DQNAgent:
    def __init__(self, observation_shape, num_actions, device='cuda:0',
                 gamma=0.99, learning_rate=0.001, weight_decay=0.0,
                 clip_gradient=True, optim_name='Adam', huber_loss=False):

        self.num_actions = num_actions
        self.gamma = gamma
        self.device = device

        self.huber_loss = huber_loss
        self.clip_gradient = clip_gradient
        self.optim_name = optim_name
        self.weight_decay = weight_decay

        self.model = VoxelDQN(observation_shape, num_actions).to(device)
        if optim_name == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=learning_rate,
                                       weight_decay=weight_decay)
        elif optim_name == "RMSProp":
            self.optimizer = optim.RMSProp(self.model.parameters(),
                                           lr=learning_rate,
                                           weight_decay=weight_decay)
        elif optim_name == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def act(self, state, mask, epsilon=0.05):
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            mask = torch.tensor(mask, dtype=torch.float).to(self.device)
            q_value = self.model(state)
            q_value = F.softmax(q_value)
            q_value *= mask
            action = q_value.max(1)[1].item()
            # print("Action: ", action)
        else:
            action = np.random.choice(np.arange(self.num_actions)[mask])
            # print("Action: ", action, "(random)")
        return action

    def compute_td_loss(self, state, action, reward, next_state, done, mask):
        state      = state.to(self.device)
        action     = action.to(self.device)
        reward     = reward.to(self.device)
        next_state = next_state.to(self.device)
        done       = done.to(self.device)
        mask       = mask.to(self.device)

        q_values      = self.model(state)
        next_q_values = self.model(next_state)
        next_q_values *= mask

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        if self.huber_loss:
            loss = huber_loss(q_value, expected_q_value.detach(), delta=10.0)
        else:
            loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_gradient:
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)

        if self.optim_name == 'Adam':
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-self.weight_decay * group['lr'], param.data)
        self.optimizer.step()
        return loss.item()

    def load_weights(self, model_path):
        if model_path is None: return
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, output, tag=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, tag))


class DDQNAgent:
    def __init__(self, observation_shape, num_actions, device='cuda:0',
                 gamma=0.99, learning_rate=0.001, weight_decay=0.0,
                 update_tar_interval=1000, clip_gradient=True, optim_name='Adam'):

        self.num_actions = num_actions
        self.gamma = gamma
        self.device = device

        self.clip_gradient = clip_gradient
        self.optim_name = optim_name
        self.weight_decay = weight_decay

        self.update_tar_interval = update_tar_interval

        self.model = VoxelDQN(observation_shape, num_actions).to(device)
        self.target_model = VoxelDQN(observation_shape, num_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        if optim_name == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=learning_rate,
                                       weight_decay=weight_decay)
        elif optim_name == "RMSProp":
            self.optimizer = optim.RMSProp(self.model.parameters(),
                                           lr=learning_rate,
                                           weight_decay=weight_decay)
        elif optim_name == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def act(self, state, mask, epsilon=0.05):
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            mask = torch.tensor(mask, dtype=torch.float).to(self.device)
            q_value = self.model(state)
            q_value = F.softmax(q_value)
            q_value *= mask
            action = q_value.max(1)[1].item()
            print("Action: ", action)
        else:
            action = np.random.choice(np.arange(self.num_actions)[mask])
            print("Action: ", action, "(random)")
        return action

    def compute_td_loss(self, state, action, reward, next_state, done, mask, frame_idx):
        state      = state.to(self.device)
        action     = action.to(self.device)
        reward     = reward.to(self.device)
        next_state = next_state.to(self.device)
        done       = done.to(self.device)
        mask       = mask.to(self.device)

        q_values      = self.model(state)
        next_q_values = self.model(next_state)
        next_q_values *= mask
        next_q_state_values = self.target_model(next_state)
        next_q_state_values *= mask

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_gradient:
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)

        if self.optim_name == 'Adam':
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-self.weight_decay * group['lr'], param.data)
        self.optimizer.step()

        if frame_idx % self.update_tar_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()


    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)
        frame_idx = checkpoint['frames']
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['model'])
        return frame_idx

    def save_model(self, output, tag=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, tag))


def huber_loss(q_value, expected_q_value, delta=1.0):
    error = q_value - expected_q_value
    quadratic_term = error * error / 2
    linear_term = torch.abs(error) - 0.5
    use_linear_term = (torch.abs(error) > delta).float()
    loss = use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term
    return loss.mean()

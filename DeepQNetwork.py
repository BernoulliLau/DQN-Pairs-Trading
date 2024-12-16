import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128,64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 32)
        self.ln3 = nn.LayerNorm(32)
        self.out = nn.Linear(32, action_size)

    def forward(self,x):
        x = x.squeeze(1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.ln3(x)
        x = F.relu(x)
        x = self.out(x)
        return x

class Agent:
    def __init__(self, model, memory, cfg, is_eval=False):
        self.n_actions = cfg['n_actions']
        self.device = torch.device(cfg['device'])
        self.gamma = cfg['gamma']

        self.sample_count = 0
        self.epsilon = cfg['epsilon']
        self.epsilon_end = cfg['epsilon_end']
        self.epsilon_decay = cfg['epsilon_decay']
        self.batch_size = cfg['batch_size']
        self.policy_net = model.to(self.device)
        self.target_net = model.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.is_eval = is_eval
        self.target_net.eval() if is_eval else self.target_net.train()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg['lr'])
        self.memory = memory

    def action(self, state):
        """select action"""
        self.sample_count += 1
        if random.random() <= self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item()

        self.epsilon *= self.epsilon_decay
        return action

    @torch.no_grad()
    def predict_action(self,state):
        """predict action"""
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        q_values = self.policy_net(state)
        action = q_values.max(1)[1].item()

        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = tuple(zip(*transitions))
        states, actions, rewards, next_states, dones = map(torch.tensor, zip(*batch))

        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

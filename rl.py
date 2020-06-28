# https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
import copy
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from itertools import count
from collections import deque, namedtuple


Epsilon = namedtuple('Epsilon', ('start', 'end', 'decay'))


class Trigno(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.cat([x, torch.sin(x), torch.cos(x)], -1)
        return x


class Outer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, f = x.shape
        y = torch.einsum('bi,bj->bij', x, x)
        y = y.view(b, f*f)
        y = torch.cat([x, y], -1)
        return y


class GenericFCN(nn.Module):

    def __init__(self, n_in, n_out, layers=None):
        super().__init__()
        layers = layers or [256, 128, 64, 32]
        self.model = nn.Sequential(
            Trigno(),
            Outer(),
            nn.Linear(9*n_in*n_in+3*n_in, layers[0]),
            nn.BatchNorm1d(layers[0]),
            nn.GELU(),
            *(nn.Sequential(
                nn.Linear(l1, l2),
                nn.BatchNorm1d(l2),
                nn.GELU(),
            ) for l1, l2 in zip(layers, layers[1:])),
            nn.Linear(layers[-1], n_out),
            nn.Sigmoid()
        )

    def forward(self, *x):
        x = torch.cat(x, -1)
        return self.model(x)
    

class EMANetwork:

    def __init__(self, network, lr=1e-4):
        self.network = network.train()
        self.target = copy.deepcopy(network).eval()
        self.optim = torch.optim.Adam(self.network.parameters(), lr)

    def optimize(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.ema()

    def ema(self, mu=0.01):
        for target_param, param in zip(self.network.parameters(), self.target.parameters()):
            target_param.data.copy_(param.data * mu + target_param.data * (1.0 - mu))


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward.unsqueeze(0), next_state, torch.tensor([done]))
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class RLAI:

    def __init__(self):        
        self.device = 'cuda'
        self.num_state = 24
        self.num_action = 8

        self.replay = Memory(1000000)
        self.actor = GenericFCN(self.num_state, self.num_action)
        self.critic = GenericFCN(self.num_state + self.num_action, self.num_action)
        
        self.actor = EMANetwork(self.actor.to(self.device))
        self.critic = EMANetwork(self.critic.to(self.device))
        
        self.loss_fn = nn.MSELoss()
        self.eps = Epsilon(0.9, 0.05, 10000)
        self.batch_size = 4096
        self.gamma = 0.99
        self.steps = 0
        self.loss = 0

    @torch.no_grad()
    def _sample_action(self, state):
        self.actor.network.eval()
        action = self.actor.network(state)[0]
        self.action = action.unsqueeze(0)
        return self.action

    def _random_action(self):
        action = torch.rand(self.num_action)
        self.action = action.unsqueeze(0).to(self.device)
        return self.action

    def next_action(self, state):
        eps = self.eps
        sample = random.random()
        eps_threshold = eps.end + (eps.start - eps.end) * \
            math.exp(-1. * self.steps / eps.decay)
        self.steps += 1
        if sample > eps_threshold:
            return self._sample_action(state)
        return self._random_action()

    def optimize_step(self):
        if len(self.replay) < self.batch_size:
            return
        
        self.actor.network.train()
        states, actions, rewards, next_states, _ = \
            map(torch.cat, self.replay.sample(self.batch_size))

        Qvals = self.critic.network(states, actions)
        next_actions = self.actor.target(next_states)
        next_Q = self.critic.target(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q

        critic_loss = self.loss_fn(Qvals, Qprime)
        policy_loss = -self.critic.network(states, self.actor.network(states)).mean()

        self.actor.optimize(policy_loss)
        self.critic.optimize(critic_loss)

    def init_run(self, state):
        state = state.to(self.device)
        self.last_state = state
        return self._sample_action(state)

    def run_step(self, state, reward, done):
        state = state.to(self.device)
        reward = torch.tensor([reward]).to(self.device)

        if not done:
            self.replay.push(self.last_state, self.action, reward, state, done)
            self.optimize_step()
            self.last_state = state

        return self.next_action(state)
        
    def save(self, fn):
        torch.save(self.actor.network.state_dict(), fn)

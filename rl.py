import math
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from itertools import count
from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
Epsilon = namedtuple('Epsilon', 
                        ('start', 'end', 'decay'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, num_stats, num_controls):
        super(DQN, self).__init__()
        layers = [32, 64, 128, 64, 32]
        self.model = nn.Sequential(
            nn.Linear(num_stats, layers[0]),
            nn.ReLU(),
            *(nn.Sequential(
                nn.Linear(l1, l2),
                nn.ReLU(),
            ) for l1, l2 in zip(layers, layers[1:])),
            nn.Linear(layers[-1], num_controls),
        )

    def forward(self, x):
        return self.model(x)
    

class RLAI:

    def __init__(self):        
        self.device = 'cuda'
        self.num_state = 24
        self.num_action = 8

        self.replay = ReplayMemory(10000)
        self.policy = DQN(self.num_state, self.num_action).to(self.device)
        self.target = DQN(self.num_state, self.num_action).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        
        self.optim = optim.RMSprop(self.policy.parameters())
        self.eps = Epsilon(0.9, 0.05, 200)
        self.batch_size = 256
        self.gamma = 0.999
        self.steps = 0
        self.loss = 0

    @torch.no_grad()
    def _sample_action(self, state):
        action = self.policy(state)[0]
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

        batch = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*batch))

        non_final_mask = map(lambda s: s is not None, batch.next_state)
        non_final_mask = torch.tensor(tuple(non_final_mask)).bool().to(self.device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # RE: Understand what happens below 
        state_action_values = self.policy(state_batch).gather(1, (action_batch>0.5).long())
        print(state_action_values.shape)
        next_state_values = torch.zeros(self.batch_size).to(self.device)
        print(next_state_values.shape, non_final_mask.shape, non_final_next_states.shape)
        next_state_values[non_final_mask] = self.target(non_final_next_states)[0].detach()
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # End RE:

        loss = F.binary_cross_entropy(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optim.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

        self.loss = loss.item()

    def init_run(self, state):
        state = state.to(self.device)
        self.last_state = state
        return self._sample_action(state)

    def run_step(self, state, reward, done):
        state = state.to(self.device)
        reward = torch.tensor([reward]).to(self.device)

        if done:
            state = None

        self.replay.push(self.last_state, self.action, state, reward)
        self.optimize_step()

        self.last_state = state
        return self.next_action(state)
        

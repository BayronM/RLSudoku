import numpy as np
import math
import random
import matplotlib.pyplot as plt
from collections import deque, namedtuple

import gymnasium as gym
from gymnasium import spaces

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(n_observations, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class DQNAgent():
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 1000
        self.batch_size = 64
        self.tau = 0.005
        self.lr = 0.001

        self.n_observations = 81
        self.n_actions = 729

        self.policy_net = DQNetwork(self.n_observations, self.n_actions).to(device)
        self.target_net = DQNetwork(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.episode_rewards = []

    def select_action(self, state):
        self.steps_done += 1
        sample = random.random()
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        if sample > epsilon:
            # Exploit
            with torch.no_grad():
                action_values = self.policy_net(state)
                action = np.unravel_index(action_values.view(9,9,9).argmax().item(), (9,9,9))
                # +1 because the action space is 1-9
                return action[0], action[1], action[2]
        else:
            # Explore
            return (random.randrange(9), random.randrange(9), random.randrange(9))

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).view(-1,1)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        creiterion = nn.SmoothL1Loss()
        loss = creiterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        
    def train(self, num_episodes=100):
        for i_episode in range(num_episodes):
            print("Episode: ", i_episode)
            state = torch.tensor(self.env.reset(), device=device, dtype=torch.float)
            state = state.view(1, 9*9)  # Reshape the state tensor
            done = False
            step = 0
            while not done and step < 1000:
                # select and perform action
                while True:
                    action = self.select_action(state)
                    action_value = action[2]+1
                    next_state, reward, done, _ = self.env.step((action[0], action[1], action_value))
                    
                    #only choose empty cells to fill
                    if next_state[action[0]][action[1]] == 0:
                        break
                print(f"action: {action}, reward: {reward}")
                # observe new state
                next_state = torch.tensor(next_state, device=device, dtype=torch.float)
                next_state = next_state.view(1, 9*9)  # Reshape the next_state tensor

                # Store the transition in memory
                reward = torch.tensor([reward], device=device)
                action = torch.tensor([np.ravel_multi_index(action, (9,9,9))], device=device)  # Reshape action to 1-D

                self.memory.push(state, action, next_state, reward)
                # move to the next state
                state = next_state
                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                step += 1
                if done:
                    print(f"solution found in {step} steps")
                    self.env.render()
                    break
            # Update the target network
            if i_episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print("Target network updated")
    
    def test(self,episodes):
        for i_episode in range(episodes):
            state = torch.tensor(self.env.reset(), device=device, dtype=torch.float)
            state = state.view(1, 9*9)
            for t in range(1000):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.tensor(next_state, device=device, dtype=torch.float)
                next_state = next_state.view(1, 9*9)
                state = next_state
                if done:
                    break
            print("Episode: ", i_episode, "Score: ", t)




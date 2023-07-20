import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device("cuda")

class ActorCritic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class PPOAgent():
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99
        self.clip_epsilon = 0.2
        self.n_observations = 81
        self.n_actions = 729
        self.model = ActorCritic(self.n_observations, self.n_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.entropy_coef = 0.01  
        

    def select_action(self, state):
        with torch.no_grad():
            policy, _ = self.model(state)
            probs = policy.detach().cpu().numpy().squeeze()
            action = np.random.choice(self.n_actions, p=probs)
        return action, policy[0][action]

    def train(self, num_episodes=100):
        done = False
        total_rewards = []  # List to store total rewards per episode
        total_steps = []  # List to store total steps per episode
        for i_episode in range(num_episodes):
            state = torch.tensor(self.env.reset(), device=device, dtype=torch.float)
            state = state.view(1, 9*9)
            done = False
            step = 0
            total_reward = 0  # Reset total reward for the episode
            while not done and step < 162:
                action, new_prob = self.select_action(state)
                action_unraveled = np.unravel_index(action, (9,9,9))
                action_value = action_unraveled[2]+1
                next_state, reward, done, _ = self.env.step((action_unraveled[0], action_unraveled[1], action_value))
                total_reward += reward  # Add reward to total reward
                next_state = torch.tensor(next_state, device=device, dtype=torch.float)
                next_state = next_state.view(1, 9*9)
                reward = torch.tensor([reward], device=device)

                old_value = self.model.critic(state)
                new_value = self.model.critic(next_state)
                advantage = reward + self.gamma * new_value - old_value
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # Normalization of advantages
                ratio = new_prob / new_prob.detach()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage
                entropy = -new_prob * torch.log(new_prob + 1e-8)  # Entropy term
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()  # Entropy-augmented loss
                value_loss = F.smooth_l1_loss(old_value, reward + self.gamma * new_value.detach())
                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                state = next_state
                step += 1
                if done:
                    print(f"Episode: {i_episode}, solution found in {step} steps, total reward: {total_reward}")
                    self.env.render()

                    total_rewards.append(total_reward)  # Append total reward for the episode
                    total_steps.append(step)  # Append total steps for the episode
                    break
            else:
                print(f"Episode: {i_episode}, no solution found within 162 steps, total reward: {total_reward}")
                total_rewards.append(total_reward)  # Append total reward for the episode
                total_steps.append(step)  # Append total steps for the episode
            if i_episode % 10 : total_rewards.append(total_reward)  # Append total reward for the episode
            if i_episode % 1000 == 0:
                self.save_checkpoint(filename=f"checkpoint_{i_episode}.pth.tar")
                plt.figure(figsize=(10, 5))
                plt.plot(total_rewards)
                plt.xlabel('Episode')
                plt.ylabel('Total reward')
                plt.title('Total reward per episode')
                plt.savefig(f'img/total_reward_episode_{i_episode}.png')
                
        print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards)}")
        print(f"Average steps over {num_episodes} episodes: {np.mean(total_steps)}")
        


    def save_checkpoint(self, filename='checkpoint.pth.tar'):
        torch.save(self.model.state_dict(), filename)

    def resume_training_from_checkpoint(self, filename='checkpoint.pth.tar', num_episodes=100):
        self.model.load_state_dict(torch.load(filename))
        self.train(num_episodes=num_episodes)

    

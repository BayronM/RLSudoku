import numpy as np
from gymnasium import spaces

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = {}
        self.alpha = 0.5  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def get_state_key(self, state):
        return str(state)

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            # Explore action space
            while True:
                action = spaces.Discrete(9).sample(), spaces.Discrete(9).sample(), spaces.Discrete(9).sample()+1
                # only choose valid actions, i.e., empty cells
                if state[action[0]][action[1]] == 0:
                    break
        else:
            # Exploit learned values
            state_key = self.get_state_key(state)
            if state_key not in self.q_table:
                return spaces.Discrete(9).sample(), spaces.Discrete(9).sample(), spaces.Discrete(9).sample()+1
            action_index = max(self.q_table[state_key], key=self.q_table[state_key].get)
            action = tuple(i.item() for i in action_index)
        return action

    def update_q_table(self, state, action, reward, new_state):
        old_state_key = self.get_state_key(state)
        new_state_key = self.get_state_key(new_state)
        if old_state_key not in self.q_table:
            self.q_table[old_state_key] = {}
        if new_state_key not in self.q_table:
            self.q_table[new_state_key] = {}

        old_value = self.q_table[old_state_key].get(action, 0)
        next_max = max(self.q_table[new_state_key].values(), default=0)

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[old_state_key][action] = new_value

    def train(self, episodes):
        for episode in range(episodes):
            print(f"Episode {episode + 1} of {episodes}")
            state = self.env.reset()
            done = False
            step = 0
            while not done and step < 1000:
                action = self.choose_action(state)
                new_state, reward, done, info = self.env.step(action,step)
                self.update_q_table(state, action, reward, new_state)
                state = new_state
                step += 1
                if done:
                    print(f"solution found in {step} steps")
                    self.env.render()
                    break




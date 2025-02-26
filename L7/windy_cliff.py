#some assistance from chat gpt
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

np.random.seed(0)

class WindyCliffWorld(gym.Env):
    def __init__(self):
        super(WindyCliffWorld, self).__init__()
        
        self.grid_size = (7, 10)
        self.start_state = (3, 0)
        self.goal_state = (3, 9)
        self.cliff = [(3, i) for i in range(1, 9)]
        self.obstacles = [(2, 4), (4, 4), (2, 7), (4, 7)]
        
        self.wind_strength = {
            (i, j): np.random.choice([-1, 0, 1]) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])
        }

        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Discrete(self.grid_size[0] * self.grid_size[1])
        
        self.state = self.start_state
        
        self.action_effects = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }

    def reset(self):
        self.state = self.start_state
        return self.state_to_index(self.state)
    
    def step(self, action):
        new_state = (self.state[0] + self.action_effects[action][0], self.state[1] + self.action_effects[action][1])
        new_state = (max(0, min(new_state[0], self.grid_size[0] - 1)), max(0, min(new_state[1], self.grid_size[1] - 1)))

        # Apply wind effect
        wind = self.wind_strength[new_state]
        new_state = (new_state[0] + wind, new_state[1])
        new_state = (max(0, min(new_state[0], self.grid_size[0] - 1)), max(0, min(new_state[1], self.grid_size[1] - 1)))

        if new_state in self.cliff:
            reward = -100
            done = True
            new_state = self.start_state
        elif new_state == self.goal_state:
            reward = 10
            done = True
        elif new_state in self.obstacles:
            reward = -10
            done = False
        else:
            reward = -1
            done = False

        self.state = new_state
        return self.state_to_index(new_state), reward, done, {}
    
    def state_to_index(self, state):
        return state[0] * self.grid_size[1] + state[1]
    
    def index_to_state(self, index):
        return (index // self.grid_size[1], index % self.grid_size[1])
    
    def render(self):
        grid = np.zeros(self.grid_size)
        grid[self.state] = 1
        for c in self.cliff:
            grid[c] = -1  
        for o in self.obstacles:
            grid[o] = -0.5  
        grid[self.goal_state] = 2  
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='viridis')
        plt.axis('off')
        fig.canvas.draw()
        plt.close(fig)
        return np.array(fig.canvas.renderer.buffer_rgba())

def q_learning(env, num_episodes, alpha, gamma, epsilon):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0  

        while not done and steps < 1000:  
            action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[state])
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            best_next_action = np.max(q_table[next_state])
            q_table[state, action] += alpha * (reward + gamma * best_next_action - q_table[state, action])

            state = next_state
            steps += 1  

        rewards_per_episode.append(total_reward)

    return q_table, rewards_per_episode

def sarsa(env, num_episodes, alpha, gamma, epsilon):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[state])
        steps = 0  

        while not done and steps < 1000:  
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            next_action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[next_state])
            q_table[state, action] += alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])

            state, action = next_state, next_action
            steps += 1  

        rewards_per_episode.append(total_reward)

    return q_table, rewards_per_episode

def visualize_policy(env, q_table, filename):
    state = env.reset()
    frames = []
    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, _, done, _ = env.step(action)
        frames.append(env.render())

    imageio.mimsave(os.path.join('./', filename), frames, duration=0.5)

env = WindyCliffWorld()
q_table, _ = q_learning(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
visualize_policy(env, q_table, filename='q_learning_windy_cliff.gif')

q_table, _ = sarsa(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
visualize_policy(env, q_table, filename='sarsa_windy_cliff.gif')

def plot_learning_curve(rewards_dict, title, filename):
    plt.figure(figsize=(10, 5))
    
    for key, rewards in rewards_dict.items():
        plt.plot(rewards, label=key)
    
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.show()

alphas = [0.1, 0.5]
epsilons = [0.1, 0.5]
num_episodes = 500
gamma = 0.99

q_rewards_dict = {}
for alpha in alphas:
    for epsilon in epsilons:
        _, rewards = q_learning(env, num_episodes, alpha, gamma, epsilon)
        q_rewards_dict[f"α={alpha}, ε={epsilon}"] = rewards

sarsa_rewards_dict = {}
for alpha in alphas:
    for epsilon in epsilons:
        _, rewards = sarsa(env, num_episodes, alpha, gamma, epsilon)
        sarsa_rewards_dict[f"α={alpha}, ε={epsilon}"] = rewards

# plot
plot_learning_curve(q_rewards_dict, "Q-Learning Performance", "q_learning_windy_cliff_hyperparameters.png")
plot_learning_curve(sarsa_rewards_dict, "SARSA Performance", "sarsa_windy_cliff_hyperparameters.png")


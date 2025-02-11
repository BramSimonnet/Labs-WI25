import numpy as np

import matplotlib.pyplot as plt
from src.maze_env import MazeEnv
from typing import List, Callable, Tuple
import matplotlib.animation as animation

# Define the states and possible actions
states = np.arange(1, 6)  # States 1 through 5
actions = ['left', 'stay', 'right']  # Available actions in each state


def transition(state, action):
    """
    Transition function that determines the next state based on the current state and action.

    Parameters:
    state (int): The current state.
    action (str): The action chosen.

    Returns:
    int: The next state.
    """
    if action == "stay":
            return state
    if action == "left":
        if state in [2,3,4,5]:
            state = state - 1
    if action == 'right': 
        if state in [1,2,3,4]:
            state += 1
    return state


def reward(state, action):
    """
    Calculate the reward for a given state and action.

    Parameters:
    state (int): The current state.
    action (str): The action taken.

    Returns:
    int: The reward.
    """
    # TODO: Your code here
    if state == 5:
        return 10  # Reward for reaching the goal
    else:
        # Small penalty for each step to encourage faster goal-reaching
        return -1  # Negative reward for every step taken




def always_right_policy(state):
    """
    Policy that always returns 'right' for any given state.

    Parameters:
    state (int): The current state.

    Returns:
    str: The chosen action ('right').
    """
    return 'right'


def my_policy(state):
    """
    This function implements a custom policy.

    Parameters:
    state (int): The current state of the system.

    Returns:
    str: The action chosen by the policy.
    """
    # TODO: Your code here
    if state == 5:
        return 'stay'  # Stay at the goal
    else:
        return 'right' 

        
def simulate_mdp(policy: Callable, initial_state=1, simulation_depth=20):
    """
    Simulates the Markov Decision Process (MDP) based on the given policy. 
    If we reach the terminal state, the simulation ends.
    Keeps track of the number of visits to each state, the cumulative reward, and the history of visited states.

    Parameters:
    - policy: A function that takes the current state as input and returns an action.
    - initial_state: The initial state of the MDP. Default is 1.
    - simulation_depth: The maximum number of steps to simulate. Default is 20.

    Returns:
    - state_visits: An array that tracks the number of visits to each state.
    - cumulative_reward: The cumulative reward obtained during the simulation.
    - visited_history: A list that tracks the history of visited states.
    - reward_history: A list that tracks the history of rewards obtained.
    """
    current_state = initial_state
    cumulative_reward = 0
    state_visits = np.zeros(len(states)) # Track the number of visits to each state
    visited_history = [current_state] # Track the history of visited states
    reward_history = [0] # Track the history of rewards
    
    for _ in range(simulation_depth):
        # TODO: Your code here
        current_state = initial_state
    cumulative_reward = 0
    state_visits = np.zeros(len(states))  # Track the number of visits to each state
    visited_history = [current_state]  # Track the history of visited states
    reward_history = [0]  # Track the history of rewards
    
    for _ in range(simulation_depth):
        action = policy(current_state)  # Get the action from the policy
        next_state = transition(current_state, action)  # Get the next state based on the action
        reward = reward(current_state, action)  # Calculate the reward
        
        # Update state visits
        state_visits[next_state - 1] += 1  # Increment the visit count for the next state
        visited_history.append(next_state)  # Add the next state to the history
        reward_history.append(reward)  # Add the reward to the history
        
        # Update cumulative reward
        cumulative_reward += reward
        
        # Update current state
        current_state = next_state
        
        # Check if terminal state is reached (assuming state 5 is terminal)
        if current_state == 5:
            break

    
    return state_visits, cumulative_reward, visited_history, reward_history


def new_policy(state: List[int]) -> int:
    # TODO: Your code here
    if 1 in state:
        return 2  # Move right from state 1
    elif 5 in state:
        return 1  # Stay in state 5 (terminal state)
    else:
        return 0  # Move left otherwise

        
def simulate_maze_env(env: MazeEnv, policy: Callable, num_steps=20):
    """
    Simulates the environment using the given policy for a specified number of steps.

    Parameters:
    - env: The environment to simulate.
    - policy: The policy to use for selecting actions (this is a function that takes a state as input and returns an action)
    - num_steps: The number of steps to simulate (default: 20).

    Returns:
    - path: The sequence of states visited during the simulation.
    - total_reward: The total reward accumulated during the simulation.
    """
    state = env.reset()
    total_reward = 0
    path = [state]

    for _ in range(num_steps):
        # TODO: Your code here
        state = env.reset()
    total_reward = 0
    path = [state]

    for _ in range(num_steps):
        action = policy(state)  # Get the action from the policy based on the current state
        next_state, reward, done, _ = env.step(action)  # Take a step in the environment
        
        total_reward += reward  # Add the reward from the step to the total reward
        path.append(next_state)  # Add the new state to the path
        
        if done:  # If the goal is reached, terminate the simulation
            break

        state = next_state  # Upd


    return path, total_reward


def q_learning(env: MazeEnv, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1) -> np.ndarray:
    """
    Perform Q-learning to learn the optimal policy for the given environment.

    Args:
        env (MazeEnv): The environment to learn the policy for.
        episodes (int, optional): Number of episodes for training. Defaults to 500.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.99.
        epsilon (float, optional): Exploration rate. Defaults to 0.1.

    Returns:
        np.ndarray: The learned Q-table.
    """
    q_table = np.zeros((env.size, env.size, env.action_space.n)) # TODO: Your code here


    for episode in range(episodes):
        # TODO: Your code here
        state = env.reset()  # Reset the environment to start a new episode
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space.n)  # Random action (exploration)
            else:
                action = np.argmax(q_table[state[0], state[1]])  # Action with highest Q-value (exploitation)
            
            # Take action and observe next state and reward
            next_state, reward, done, _ = env.step(action)
            
            # Update Q-value using the Q-learning update rule
            best_next_action = np.argmax(q_table[next_state[0], next_state[1]])  # Best action in next state
            q_table[state[0], state[1], action] += alpha * (
                reward + gamma * q_table[next_state[0], next_state[1], best_next_action] - q_table[state[0], state[1], action]
            )
            
            state = next_state


    return q_table


def simulate_maze_env_q_learning(
    env: MazeEnv, q_table: np.ndarray
) -> Tuple[List[Tuple[int, int]], bool]:
    """
    Simulate the maze environment using the Q-table to determine the actions to take.
    Also creates an animation of the agent moving through the environment.
    
    Args:
        env (MazeEnv): The maze environment instance.
        q_table (np.ndarray): The Q-table containing action values.

    Returns:
        Tuple[List[Tuple[int, int]], bool]: A tuple containing a list of states and a boolean indicating if the episode is done.
    """

    state = env.reset()
    done = False

    starting_frame = env.render(mode="rgb_array").T
    frames = [starting_frame]  # List to store frames for animation
    states = [state]  # List to store states

    while not done:
        action = ... # TODO: Your code here
        state, _, done, _ = env.step(action)
        frames.append(
            env.render(mode="rgb_array").T
        )  # Render the environment as an RGB array
        states.append(state)

    def update_frame(i):
        ax.clear()
        ax.imshow(frames[i], cmap="viridis", origin="lower")
        ax.set_title(f"Step {i+1}")
        ax.grid("on")

    # Create animation from frames
    fig, ax = plt.subplots()
    anim = animation.FuncAnimation(fig, update_frame, frames=len(frames), interval=500)
    anim.save("mdp_q_learning.gif", writer="pillow")
    return states, done
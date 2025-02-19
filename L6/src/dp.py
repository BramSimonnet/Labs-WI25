#some assistance from chat gpt

from tqdm import tqdm
from .blackjack import BlackjackEnv, Hand, Card
from collections import defaultdict


ACTIONS = ['hit', 'stick']

def policy_evaluation(env, V, policy, episodes=500000, gamma=1.0):
    """
    Monte Carlo policy evaluation:
    - Generate episodes using the current policy
    - Update state value function as an average return
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int) 

    for _ in tqdm(range(episodes), desc="Policy evaluation"):
        episode = []
        state = env.reset()
        done = False

        while not done:
            action = policy[state] 
            next_state, reward, done = env.step(action)
            episode.append((state, reward))
            state = next_state

        visited_states = set()
        G = 0  

        for t in reversed(range(len(episode))):  
            state, reward = episode[t]
            G = gamma * G + reward 

            if state not in visited_states:
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]  
                visited_states.add(state)  

    return V


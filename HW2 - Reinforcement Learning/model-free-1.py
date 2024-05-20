import gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, state_size, action_size, discount_rate=0.9, learning_rate=0.1, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        target = reward + self.discount_rate * np.max(self.q_table[next_state]) * (not done)
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def discretize_state(env, state, bins=20):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / bins
    state_adj = (state[0] - env_low) / env_dx
    state_adj = np.clip(state_adj, 0, bins-1).astype(int)
    state_index = state_adj[0] * bins + state_adj[1]
    return state_index


def train_agent(env, agent, episodes=2000, bins=20, algorithm='Q-Learning'):
    rewards = []
    for episode in range(episodes):
        current_state = discretize_state(env, env.reset(), bins)
        total_reward = 0
        done = False
        if algorithm == 'SARSA':
            current_action = agent.choose_action(current_state)  # Choose initial action for SARSA

        while not done:
            if algorithm == 'Q-Learning':
                current_action = agent.choose_action(current_state)  # Choose action for Q-Learning
            next_state, reward, terminated, truncated, info = env.step(current_action)
            next_state = discretize_state(env, next_state, bins)
            done = terminated or truncated

            if algorithm == 'SARSA':
                next_action = agent.choose_action(next_state)  # Choose next action for SARSA
                agent.learn(current_state, current_action, reward, next_state, next_action, done)
            else:  # Q-Learning
                agent.learn(current_state, current_action, reward, next_state, done)

            current_state = next_state
            if algorithm == 'SARSA':
                current_action = next_action  # Update action for SARSA

            total_reward += reward

        rewards.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon}")
    return rewards


class SARSAAgent(QLearningAgent):
    def learn(self, state, action, reward, next_state, next_action, done):
        target = reward + self.discount_rate * self.q_table[next_state, next_action] * (not done)
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])


env = gym.make('MountainCar-v0')
state_size = 20 ** len(env.observation_space.high)
action_size = env.action_space.n

# Q-Learning: Running the training process 20 times
q_learning_all_rewards = np.zeros(2000)  # To store the sum of rewards for each episode across all runs
runs = 20
for run in range(runs):
    q_learning_agent = QLearningAgent(state_size, action_size)
    q_learning_rewards = train_agent(env, q_learning_agent, episodes=2000, bins=20, algorithm='Q-Learning')
    q_learning_all_rewards += np.array(q_learning_rewards)  # Sum up rewards for each episode

# Calculate the average reward per episode across all runs for Q-Learning
q_learning_average_rewards = q_learning_all_rewards / runs

# Plotting for Q-Learning
plt.figure(figsize=(10, 6))
plt.plot(q_learning_average_rewards, label='Average Q-Learning')
plt.legend()
plt.title('Average Cumulative Reward over 20 Runs - Q-Learning')
plt.xlabel('Episode')
plt.ylabel('Average Cumulative Reward')
plt.show()

# SARSA: Running the training process 20 times
sarsa_all_rewards = np.zeros(2000)  # To store the sum of rewards for each episode across all runs
for run in range(runs):
    sarsa_agent = SARSAAgent(state_size, action_size)
    sarsa_rewards = train_agent(env, sarsa_agent, episodes=2000, bins=20, algorithm='SARSA')
    sarsa_all_rewards += np.array(sarsa_rewards)  # Sum up rewards for each episode

# Calculate the average reward per episode across all runs for SARSA
sarsa_average_rewards = sarsa_all_rewards / runs

# Plotting for SARSA
plt.figure(figsize=(10, 6))
plt.plot(sarsa_average_rewards, label='Average SARSA')
plt.legend()
plt.title('Average Cumulative Reward over 20 Runs - SARSA')
plt.xlabel('Episode')
plt.ylabel('Average Cumulative Reward')
plt.show()

import numpy as np
import gym

# Environment setup
env = gym.make('MountainCar-v0')

# Discretization parameters
POSITION_BINS = 20
VELOCITY_BINS = 20
DISCOUNT_FACTOR = 0.9

# Discretize the state space
def discretize_state(state, position_bins=POSITION_BINS, velocity_bins=VELOCITY_BINS):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / np.array([position_bins, velocity_bins])
    state_adj = (state[0] - env_low) / env_dx
    state_adj = np.clip(state_adj, 0, [position_bins-1, velocity_bins-1]).astype(int)
    return tuple(state_adj)

# Initialize Q-table
q_table = np.zeros((POSITION_BINS, VELOCITY_BINS, env.action_space.n))

# Value iteration algorithm with logging
def value_iteration(env, episodes=10000, discount_factor=0.9, log_interval=1000):
    total_steps = []
    for episode in range(episodes):
        state = discretize_state(env.reset())
        terminated, truncated = False, False
        steps = 0
        while not terminated and not truncated:
            action = np.random.choice(env.action_space.n)
            next_state_raw, _, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state_raw)

            # Simulate the transition and update the Q-table
            reward = -1  # Assume a constant reward model for simplicity
            best_future_q = np.max(q_table[next_state])
            q_table[state][action] += (reward + discount_factor * best_future_q - q_table[state][action])

            state = next_state
            steps += 1
        total_steps.append(steps)
        if (episode + 1) % log_interval == 0:
            avg_steps = np.mean(total_steps[-log_interval:])
            print(f"Episode: {episode + 1}, Average Steps: {avg_steps}")

# Extract the policy from the Q-table
def extract_policy(q_table):
    policy = np.zeros((POSITION_BINS, VELOCITY_BINS), dtype=int)
    for i in range(POSITION_BINS):
        for j in range(VELOCITY_BINS):
            policy[i, j] = np.argmax(q_table[i, j])
    return policy

# Run value iteration
value_iteration(env)

# Extract and print the optimal policy
optimal_policy = extract_policy(q_table)
print("Optimal Policy (Discretized):")
print(optimal_policy)

import random
import gym
import numpy as np

INFINITY = float('inf')


class PongAgent:
    def __init__(self):
        self.is_testing = False
        self.minimum_epsilon = 0.1
        self.environment = gym.make('Pong-v0')
        self.last_ball_position = (0, 0)

    def log_performance_metrics(self, scores, total_episodes, epsilon_value):
        avg_episode_reward = np.mean(scores)
        win_rate = sum(score > 0 for score in scores) / len(scores)
        print(f"Average Episode Reward: {avg_episode_reward:.2f}")
        print(f"Success Rate: {win_rate:.2%}")
        print(f"Final Epsilon Value (Exploration Rate): {epsilon_value:.2f}")

        # Convergence Speed
        if len(scores) >= 50:
            last_50_avg = np.mean(scores[-50:])
            print(f"Average Score of Last 50 Episodes: {last_50_avg:.2f}")
        else:
            print("Not enough episodes to calculate convergence speed.")

        # Learning Curve
        if len(scores) >= 10:
            avg_scores = []
            for i in range(0, len(scores), 10):
                avg_score = np.mean(scores[i:i + 10])
                avg_scores.append(avg_score)
            print("Learning Curve (Average Score per 10 Episodes):")
            print(avg_scores)
        else:
            print("Not enough episodes to plot learning curve.")

    def process_state(self, raw_state):
        if not isinstance(raw_state, np.ndarray):
            raw_state = np.array(raw_state)
        cropped_state = raw_state[35:195]
        downsampled_state = cropped_state[::2, ::2, 0]

        game_info = [(0,0), 0, 0, (0,0)]
        found_p1, found_p2, found_ball = False, False, False
        p1_col, p2_col = 8, 70
        p1_color, p2_color, ball_color = 213, 92, 236

        for i in range(downsampled_state.shape[0]):
            if not found_p1 and downsampled_state[i][p1_col] == p1_color:
                game_info[1], found_p1 = i, True
            if not found_p2 and downsampled_state[i][p2_col] == p2_color:
                game_info[2], found_p2 = i, True
            if not found_ball:
                for j in range(downsampled_state.shape[1]):
                    if downsampled_state[i][j] == ball_color:
                        game_info[0], found_ball = (i, j), True
                        break
            if found_p1 and found_p2 and found_ball:
                break

        if game_info[0] != (0,0) and self.last_ball_position != (0, 0):
            newX, newY = game_info[0]
            oldX, oldY = self.last_ball_position
            game_info[3] = (newX - oldX, newY - oldY)
        self.last_ball_position = game_info[0]

        return tuple(game_info)  # Convert game_info to a tuple

    def reset_environment(self):
        return self.process_state(self.environment.reset()[0])

    def choose_action_epsilon_greedy(self, Q_values, seen_states, current_state, possible_actions, epsilon):
        untried_actions = [action for action in possible_actions if (current_state, action) not in seen_states]
        if untried_actions:
            return random.choice(untried_actions)
        if random.random() < epsilon and not self.is_testing:
            return random.choice(possible_actions)
        return self.determine_best_action(Q_values, current_state, possible_actions)

    def determine_best_action(self, Q_values, state, actions):
        max_action, max_utility = None, -INFINITY
        for action in actions:
            if (state, action) not in Q_values:
                Q_values[state, action] = 0
            if Q_values[state, action] > max_utility:
                max_utility, max_action = Q_values[state, action], action
        return max_action

    def train_agent(self):
        Q_values = {}
        action_set = [0, 2, 3]
        state_action_seen = {}
        epsilon, alpha, gamma = 0.9, 0.1, 0.95
        total_episodes, epsilon_decay = 10000, 0.999
        episode_scores = []

        for episode in range(1, total_episodes + 1):
            current_state = self.reset_environment()
            epsilon = max(self.minimum_epsilon, epsilon * epsilon_decay)

            episode_score = 0
            MAX_FRAMES_PER_EPISODE = 3000
            for _ in range(MAX_FRAMES_PER_EPISODE):
                action = self.choose_action_epsilon_greedy(Q_values, state_action_seen, current_state, action_set, epsilon)
                state_action_seen[current_state, action] = True
                new_state, reward, _, _, _ = self.environment.step(action)
                episode_score += reward
                next_state = self.process_state(new_state)

                next_action = self.determine_best_action(Q_values, next_state, action_set)
                if (current_state, action) not in Q_values:
                    Q_values[current_state, action] = 0
                Q_values[current_state, action] += alpha * (reward + gamma * Q_values.get((next_state, next_action), 0) - Q_values[current_state, action])
                current_state = next_state

            episode_scores.append(episode_score)
            print(f"Episode: {episode}, Score: {episode_score}, Epsilon: {epsilon}")
            if episode % 30 == 0:  # After each 30 episodes, prints the metrics
                self.log_performance_metrics(episode_scores, episode, epsilon)


if __name__ == '__main__':
    agent = PongAgent()
    agent.train_agent()

# 03230235_frozenlake_fix.py
# Fixed Q-learning for FrozenLake


import numpy as np

def train_frozenlake(episodes=10000, alpha=0.8, gamma=0.95, epsilon=0.3):
    env = gym.make('FrozenLake-v1', is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    rewards_list = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            total_reward += reward

        rewards_list.append(total_reward)
        epsilon = max(0.01, epsilon * 0.995)

    success_rate = np.mean(rewards_list[-100:])
    print(f"Success rate (last 100 episodes): {success_rate:.2f}")
    return Q, rewards_list

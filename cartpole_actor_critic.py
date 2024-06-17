import numpy as np
import matplotlib.pyplot as plt
import gym

def fourier_basis_cartpole(state, order, bounds):
    scale = (state - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    scale = scale[:, None]
    cosines = np.cos(scale * np.arange(1, order + 1)[None, :])
    return cosines.flatten()

def normalize_state_cartpole(state, bounds):
    normalized_state = (state - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return normalized_state

def grad_ln_policy(theta, state, action):
    grad_ln_policy_matrix = np.zeros_like(theta)
    grad_ln_policy_matrix[:, action] = state
    return grad_ln_policy_matrix

def grad_state_value(state):
    return state.reshape(-1, 1)

def actor_critic_eligibility_traces(env, theta, w, num_episodes, alpha_theta, alpha_w, gamma, lambda_theta, lambda_w, epsilon, fourier_order, bounds):
    episode_returns = []
    episode_steps = []
    smoothed_returns = []
    smoothed_steps = []

    for episode in range(num_episodes):
        state = env.reset()
        state = normalize_state_cartpole(state, bounds)
        state = fourier_basis_cartpole(state, fourier_order, bounds)
        done = False
        eligibility_trace_theta = np.zeros_like(theta)
        eligibility_trace_w = np.zeros_like(w)

        episode_return = 0
        episode_step = 0

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action_probs = np.dot(state, theta)
                action_probs = np.exp(action_probs - np.max(action_probs))
                action_probs /= np.sum(action_probs)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_state, reward, done, _ = env.step(action)
            next_state = normalize_state_cartpole(next_state, bounds)
            next_state = fourier_basis_cartpole(next_state, fourier_order, bounds)

            delta = reward + gamma * np.dot(next_state, w) - np.dot(state, w)

            eligibility_trace_theta = gamma * lambda_theta * eligibility_trace_theta + grad_ln_policy(theta, state, action)
            eligibility_trace_w = gamma * lambda_w * eligibility_trace_w + grad_state_value(state)

            theta += alpha_theta * delta * eligibility_trace_theta
            w += alpha_w * delta * eligibility_trace_w

            state = next_state
            episode_return += reward
            episode_step += 1

        episode_returns.append(episode_return)
        episode_steps.append(episode_step)
        epsilon *= 0.995
        smoothed_return = np.mean(episode_returns[-50:])  # Smooth returns using a moving average
        smoothed_returns.append(smoothed_return)

        smoothed_step = np.mean(episode_steps[-50:])  # Smooth steps using a moving average
        smoothed_steps.append(smoothed_step)

        if episode % 10 == 0:
            print(f'Episode {episode}: Total Return: {episode_return}, Smoothed Return: {smoothed_return}, Steps: {episode_step}, Smoothed Steps: {smoothed_step}')

    return smoothed_returns, smoothed_steps

def evaluate_policy(env, theta, num_episodes, fourier_order, bounds):
    returns = []
    steps = []

    for _ in range(num_episodes):
        state = env.reset()
        state = normalize_state_cartpole(state, bounds)
        state = fourier_basis_cartpole(state, fourier_order, bounds)
        done = False
        episode_return = 0
        episode_steps = 0

        while not done:
            action_probs = np.dot(state, theta)
            action_probs = np.exp(action_probs - np.max(action_probs))
            action_probs /= np.sum(action_probs)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_state, reward, done, _ = env.step(action)
            next_state = normalize_state_cartpole(next_state, bounds)
            next_state = fourier_basis_cartpole(next_state, fourier_order, bounds)

            state = next_state
            episode_return += reward
            episode_steps += 1

        returns.append(episode_return)
        steps.append(episode_steps)

    return returns, steps

def main():
    env = gym.make('CartPole-v1')
    num_episodes = 1000
    num_runs = 5
    eval_episodes = 100
    fourier_order = 5
    epsilon = 0.1
    bounds = np.array(list(zip(env.observation_space.low, env.observation_space.high)))

    theta_shape = (env.observation_space.shape[0] * fourier_order, env.action_space.n)
    w_shape = (env.observation_space.shape[0] * fourier_order, 1)

    alpha_theta = 0.001
    alpha_w = 0.01
    gamma = 0.99
    lambda_theta = 0.9
    lambda_w = 0.9

    all_smoothed_returns = []
    all_smoothed_steps = []

    for run in range(num_runs):
        theta = np.random.randn(*theta_shape) * 0.001
        w = np.zeros(w_shape)

        smoothed_returns, smoothed_steps = actor_critic_eligibility_traces(env, theta, w, num_episodes, alpha_theta, alpha_w, gamma, lambda_theta, lambda_w, epsilon, fourier_order, bounds)

        all_smoothed_returns.append(smoothed_returns)
        all_smoothed_steps.append(smoothed_steps)

    # Plot Sum of Rewards (used for evaluation)
    plt.figure(figsize=(10, 5))
    avg_smoothed_returns = np.mean(all_smoothed_returns, axis=0)
    std_smoothed_returns = np.std(all_smoothed_returns, axis=0)
    plt.plot(range(1, num_episodes + 1), avg_smoothed_returns, label='Average Smoothed Return')
    plt.fill_between(range(1, num_episodes + 1), avg_smoothed_returns - std_smoothed_returns, avg_smoothed_returns + std_smoothed_returns, alpha=0.2, color='gray')
    plt.xlabel('Episode')
    plt.ylabel('Average Smoothed Episode Return')
    plt.title('Average Smoothed Sum of Rewards over Episodes (Evaluation)')
    plt.legend()
    plt.show()

    # Plot Number of Steps per Episode (used for evaluation)
    plt.figure(figsize=(10, 5))
    avg_smoothed_steps = np.mean(all_smoothed_steps, axis=0)
    std_smoothed_steps = np.std(all_smoothed_steps, axis=0)
    plt.plot(range(1, num_episodes + 1), avg_smoothed_steps, label='Average Smoothed Steps')
    plt.fill_between(range(1, num_episodes + 1), avg_smoothed_steps - std_smoothed_steps, avg_smoothed_steps + std_smoothed_steps, alpha=0.2, color='gray')
    plt.xlabel('Episode')
    plt.ylabel('Average Smoothed Steps per Episode')
    plt.title('Average Smoothed Number of Steps over Episodes (Evaluation)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


# CartPole Actor-Critic with Eligibility Traces

This project implements an actor-critic algorithm with eligibility traces to solve the CartPole-v1 environment from OpenAI Gym using Fourier basis functions.

## Description

The algorithm uses the following components:
- **Fourier Basis Functions**: To represent the state space more effectively.
- **Normalization**: To scale state features between 0 and 1.
- **Actor-Critic with Eligibility Traces**: To update both the policy (actor) and the value function (critic) using eligibility traces for better learning efficiency.

## Files

- `cartpole_actor_critic.py`: The main implementation of the actor-critic algorithm with eligibility traces.
- `README.md`: This file, providing an overview of the project.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- OpenAI Gym

## How to Run

1. Install the required packages:
   ```bash
   pip install numpy matplotlib gym

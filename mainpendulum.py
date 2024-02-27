import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensordict
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import random
import imageio
import pathlib
import os

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from itertools import count
from DDPG.networks import Actor_net, Critic_net
from DDPG.noise import ornstein_uhlenbeck
from DDPG.agent import Agent_DDPG
from record import record_image

# Setup game
env = gym.make('Pendulum-v1')

# Set random number seeds
seed_value = 965
torch.manual_seed(seed_value)  # PyTorch
np.random.seed(seed_value)     # NumPy
random.seed(seed_value)        # Python

# Setup agent
agent_pen = Agent_DDPG(state_dim = env.observation_space._shape[0],
                        action_dim = env.action_space.shape[0],
                        action_space_scale = env.action_space.high[0],
                        size_memory = 10000,
                        batch_size = 64,
                        gamma = 0.99,
                        tau = 0.001,
                        lr_actor = 1e-4,
                        lr_critic = 1e-3,
                        sigma = 0.2,
                        dt = 0.01)

# Train agent
num_episodes = 200
score = []
for i_episode in range(num_episodes):


    # Record agents behaviour (This does not form part of the training run)
    if((i_episode + 1) % 20 == 0):
        record_image(agent_env = 'Pendulum-v1', agent = agent_pen, num_iter = (i_episode + 1), example_path = 'Pendulum_results2')

    state,_ = env.reset()
    total_reward = 0
    agent_pen.noise.reset()

    for t in count():

      state = torch.tensor(state, dtype = torch.float32)
      agent_pen.actor.eval()
      with torch.no_grad():
          action = agent_pen.actor(state.view(1, -1))[0]
      action += agent_pen.noise.sample()  
      next_state, reward, terminated, truncated, _ = env.step(action.tolist())
      done = terminated or truncated

      total_reward += reward

      # Store transition pair
      agent_pen.cache(state, action, next_state, reward, done)

      # Update critic and actor
      agent_pen.update()

      # Update target networks
      agent_pen.update_target()

      # Update state
      state = next_state

      if done:
        score.append(total_reward)
        print(i_episode)
        break
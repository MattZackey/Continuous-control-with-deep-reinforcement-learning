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
env = gym.make('MountainCarContinuous-v0')

# Set random number seeds
seed_value = 965
torch.manual_seed(seed_value)  # PyTorch
np.random.seed(seed_value)     # NumPy
random.seed(seed_value)        # Python

# Set up agent
agent_car = Agent_DDPG(state_dim = env.observation_space._shape[0],
                       action_dim = env.action_space.shape[0],
                       action_space_scale = env.action_space.high[0],
                       size_memory = 1000000,
                       batch_size = 64,
                       gamma = 0.99,
                       tau = 0.001,
                       lr_actor = 1e-4,
                       lr_critic = 1e-3,
                       sigma = 0.4,
                       dt = 0.01)

# Train agent
score = []
num_episodes = 300
for i_episode in range(num_episodes):

    # Record agents behaviour (This does not form part of the training run)
    if((i_episode + 1) % 50 == 0):
        record_image(agent_env = 'MountainCarContinuous-v0', agent = agent_car, num_iter = (i_episode + 1), example_path = 'MountainCar_results' , seed = 9056)
    
    state,_ = env.reset(seed = 9056)
    total_reward = 0
    agent_car.noise.reset()

    for t in count():

      state = torch.tensor(state, dtype = torch.float32)
      agent_car.actor.eval()
      with torch.no_grad():
          action = agent_car.actor(state.view(1, -1))[0]
      action += agent_car.noise.sample()
      next_state, reward, terminated, truncated, _ = env.step(action.tolist())
      done = terminated or truncated

      total_reward += reward

      # Store transition pair
      agent_car.cache(state, action, next_state, reward, done)

      # Update critic and actor
      agent_car.update()

      # Update target networks 
      agent_car.update_target()

      # Update state
      state = next_state

      if done:
        score.append(total_reward)
        print(i_episode)
        break
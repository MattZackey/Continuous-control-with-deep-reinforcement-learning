import torch
import numpy as np

class ornstein_uhlenbeck:

    def __init__(self, action_dim, sigma, dt, theta = 0.15):

        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.action_dim = action_dim
        self.mu = np.zeros(action_dim)

    def reset(self):
      self.x_curr = np.zeros(self.action_dim)

    def sample(self):
      x_new = self.x_curr - (self.theta * (self.x_curr - self.mu) * self.dt) + (self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim))
      self.x_curr = x_new
      return torch.tensor(self.x_curr)
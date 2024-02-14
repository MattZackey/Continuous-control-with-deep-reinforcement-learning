import torch
import torch.nn as nn
import torch.nn.functional as F
import tensordict
import torch.optim as optim
import numpy as np

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from DDPG.networks import Actor_net, Critic_net
from DDPG.noise import ornstein_uhlenbeck

class Agent_Pendulum:

    def __init__(self, state_dim, action_dim, action_space_scale, size_memory, batch_size, gamma, tau, lr_actor, lr_critic):

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize Actor newtorks
        self.actor = Actor_net(self.state_dim,  action_space_scale)
        self.target_actor = Actor_net(self.state_dim,  action_space_scale)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Initialize Critic networks
        self.critic = Critic_net(self.state_dim, self.action_dim)
        self.target_critic = Critic_net(self.state_dim, self.action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.noise = ornstein_uhlenbeck(action_dim = action_dim)
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(size_memory))
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer_actor = optim.AdamW(self.actor.parameters(), lr = lr_actor, amsgrad=True)
        self.optimizer_critic = optim.AdamW(self.critic.parameters(), lr = lr_critic, amsgrad=True, weight_decay=0.01)

    # Add experience to memory
    ############################################################################
    def cache(self, state, action, next_state, reward, done):

        next_state = torch.tensor(next_state, dtype = torch.float32)
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        self.memory.add(TensorDict({"state" : state, "action" : action, "next_state" : next_state,"reward" : reward, "done" : done}, batch_size=[]))
    ############################################################################

    # Update Actor and Critic
    ############################################################################
    def update(self):

        if len(self.memory) < self.batch_size:
            return

        #Sample a batch
        self.batch = self.memory.sample(self.batch_size)

        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

        # Update Critic
        ####################################################
        pred_state_action_vals = self.critic(self.batch['state'], self.batch['action'])
        actions_prime = self.target_actor(self.batch['next_state'])
        with torch.no_grad():
            next_state_action_vals = self.target_critic(self.batch['next_state'], actions_prime) * (1 - self.batch['done'].int())
        exp_state_action_batch = self.batch['reward'] + (self.gamma * next_state_action_vals)

        # Compute Huber loss
        critic_loss = self.loss_fn(pred_state_action_vals, exp_state_action_batch)

        # Set to training
        self.critic.train()

        # Reset gradients
        self.critic.zero_grad()

        # Compute gradients
        critic_loss.backward()

        # Prevent vanishing gradients
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 5)

        # Update parameters
        self.optimizer_critic.step()

        self.critic.eval()
        ####################################################

        # Update Actor
        ####################################################
        # Set to training
        self.actor.train()

        # Reset gradients
        self.optimizer_actor.zero_grad()

        actions_pred = self.actor(self.batch['state'])
        state_actions_pred = self.critic(self.batch['state'], actions_pred)
        actor_loss = torch.mean(-state_actions_pred)
        actor_loss.backward()

        # Prevent vanishing gradients
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), 5)

        # Update parameters
        self.optimizer_actor.step()
        ####################################################

    # Update target networks
    ############################################################################
    def update_target(self):

        # Update target Critic network
        self.critic_dict = self.critic.state_dict()
        self.target_critic_dict = self.target_critic.state_dict()
        for i in self.critic_dict:
            self.target_critic_dict[i] =  (self.critic_dict[i] * self.tau) + (self.target_critic_dict[i] * (1 - self.tau))
        self.target_critic.load_state_dict(self.target_critic_dict)

        # Update target Actor network
        self.actor_dict = self.actor.state_dict()
        self.target_actor_dict = self.target_actor.state_dict()
        for i in self.actor_dict:
            self.target_actor_dict[i] =  (self.actor_dict[i] * self.tau) + (self.target_actor_dict[i] * (1 - self.tau))
        self.target_actor.load_state_dict(self.target_actor_dict)
    ############################################################################
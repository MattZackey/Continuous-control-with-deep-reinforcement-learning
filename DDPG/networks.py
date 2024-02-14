import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Actor neural network
#################################################################################
class Actor_net(nn.Module):

    def __init__(self, state_dim, action_space_scale):
        super(Actor_net, self).__init__()

        self.state_dim = state_dim
        self.action_space_scale = action_space_scale

        self.layer1 = nn.Linear(self.state_dim, 400)
        self.layer1_norm = nn.BatchNorm1d(400)
        self.layer2 = nn.Linear(400, 300)
        self.layer2_norm = nn.BatchNorm1d(300)
        self.layer3 = nn.Linear(300, 1)

        self.intial_parameters()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer1_norm(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = self.layer2_norm(x)
        x = F.relu(x)
        x = F.tanh(self.layer3(x))
        x = x * self.action_space_scale
        return x

    def intial_parameters(self):
        nn.init.uniform_(self.layer1.weight, a = -1/math.sqrt(400), b =  1/math.sqrt(400))
        nn.init.uniform_(self.layer1.bias, a = -1/math.sqrt(400), b = 1/math.sqrt(400))
        nn.init.uniform_(self.layer2.weight, a = -1/math.sqrt(300), b =  1/math.sqrt(300))
        nn.init.uniform_(self.layer2.bias, a = -1/math.sqrt(300), b = 1/math.sqrt(300))
        nn.init.uniform_(self.layer3.weight, a = -3e-03, b = 3e-03)
        nn.init.uniform_(self.layer3.bias, a = -3e-04, b = 3e-04)
#################################################################################

# Critic neural network
#################################################################################
class Critic_net(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic_net, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # State layers
        self.layer1_state = nn.Linear(self.state_dim, 400)
        self.layer1_state_norm = nn.BatchNorm1d(400)
        self.layer2_state = nn.Linear(400, 300)
        self.layer2_state_norm = nn.BatchNorm1d(300)

        # Action layers
        self.layer1_action = nn.Linear(self.action_dim, 300)

        # Combined layer
        self.layer3 = nn.Linear(300, 1)

        self.intial_parameters()

    def forward(self, state, action):

        # Forward pass for state
        state_value = self.layer1_state(state)
        state_value = self.layer1_state_norm(state_value)
        state_value = F.relu(state_value)
        state_value = self.layer2_state(state_value)
        state_value = self.layer2_state_norm(state_value)

        # Forward pass for action
        action_value = self.layer1_action(action)
        action_value = F.relu(action_value)

        # Forward pass for combined
        x = F.relu(torch.add(state_value, action_value))
        x = self.layer3(x)

        return x

    def intial_parameters(self):
        nn.init.uniform_(self.layer1_state.weight, a = -1/math.sqrt(400), b =  1/math.sqrt(400))
        nn.init.uniform_(self.layer1_state.bias, a = -1/math.sqrt(400), b = 1/math.sqrt(400))
        nn.init.uniform_(self.layer2_state.weight,  -1/math.sqrt(300), b =  1/math.sqrt(300))
        nn.init.uniform_(self.layer2_state.bias, a = -1/math.sqrt(300), b = 1/math.sqrt(300))
        nn.init.uniform_(self.layer3.weight, a = -3e-03, b =  3e-03)
        nn.init.uniform_(self.layer3.bias, a = -3e-04, b = 3e-04)
#################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class GradientPolicy(nn.Module):

  def __init__(self, hidden_size, obs_size, out_dims, max):
    super().__init__()


    self.max = torch.from_numpy(max).to(device)
    self.net = nn.Sequential(
        nn.Linear(obs_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU()
    )

    self.linear_mu = nn.Linear(hidden_size, out_dims)
    self.linear_std = nn.Linear(hidden_size, out_dims)

  def forward(self, obs):
    if isinstance(obs, np.ndarray):
      obs = torch.from_numpy(obs).to(device)

    x = self.net(obs.float())
    mu = self.linear_mu(x) #mean of actions

    std = self.linear_std(x) #std of actions
    std= F.softplus(std) + 1e-3

    dist = Normal(mu, std)
    action = dist.rsample()
    log_prob = dist.log_prob(action)
    log_prob = log_prob.sum(dim=-1, keepdim=True)
    log_prob -= (2 * (np.log(2) -action - F.softplus(-2 * action))).sum(dim=-1, keepdim=True)

    action = torch.tanh(action) * self.max
    return action, log_prob

import torch
import torch.nn as nn
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class DQN(nn.Module):

  def __init__(self, hidden_size, obs_size, out_dims):
    super().__init__()

    self.net = nn.Sequential(
        nn.Linear(obs_size + out_dims, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1)
    )

  def forward(self, state, action):
    if isinstance(state, np.ndarray):
      state = torch.from_numpy(state).to(device)
    if isinstance(action, np.ndarray):
      action = torch.from_numpy(action).to(device)
    
    in_vector = torch.hstack((state, action))
    return self.net(in_vector.float())
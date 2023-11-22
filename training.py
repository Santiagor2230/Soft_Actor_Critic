import copy
import torch
import itertools
import random

import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW
from pytorch_lightning import LightningModule

from environment import create_environment
from dqn_model import DQN
from policy_model import GradientPolicy
from replay_buffer import ReplayBuffer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def polyak_average(net, target_net, tau = 0.01):
  for qp, tp in zip(net.parameters(), target_net.parameters()):
    tp.data.copy_(tau*qp.data + (1-tau) * tp.data)

class RLDataset(IterableDataset):

  def __init__(self, buffer, sample_size= 400):
    self.buffer = buffer
    self.sample_size = sample_size
  
  def __iter__(self):
    for experience in self.buffer.sample(self.sample_size):
      yield experience
      
      
class SAC(LightningModule):

  def __init__(self, env_name, capacity=100_000, batch_size=256, lr=1e-3,
               hidden_size=256, gamma=0.99, loss_fn = F.smooth_l1_loss,
               optim=AdamW, samples_per_epoch=1_000, tau = 0.005, 
               epsilon=0.05, alpha=0.02,
               ):
    super().__init__()

    self.env = create_environment(env_name)

    obs_size = self.env.observation_space.shape[0]
    action_dims = self.env.action_space.shape[0]
    max_action = self.env.action_space.high

    self.q_net1 = DQN(hidden_size, obs_size, action_dims)
    self.q_net2 = DQN(hidden_size, obs_size, action_dims)

    self.policy = GradientPolicy(hidden_size, obs_size, action_dims, max_action)

    self.target_q_net1 = copy.deepcopy(self.q_net1)
    self.target_q_net2 = copy.deepcopy(self.q_net2)

    self.target_policy = copy.deepcopy(self.policy)

    self.buffer = ReplayBuffer(capacity = capacity)

    self.save_hyperparameters()

    while len(self.buffer) < self.hparams.samples_per_epoch:
      print(f'{len(self.buffer)} samples in experience buffer. Filling...')
      self.play_episode()

  @torch.no_grad()
  def play_episode(self, policy=None):
    obs = self.env.reset()
    done = False

    while not done:
      if policy and random.random()> self.hparams.epsilon:
        action,_ = self.policy(obs)
        action = action.cpu().numpy()
      else:
        action=self.env.action_space.sample()

      next_obs, reward, done, info = self.env.step(action)
      exp = (obs, action, reward, done, next_obs)
      self.buffer.append(exp)
      obs = next_obs

  def forward(self, x):
    output = self.policy(x)
    return output

  def configure_optimizers(self):
    '''iterate between both q networks'''
    q_net_params = itertools.chain(self.q_net1.parameters(),self.q_net2.parameters())

    '''set as paramater the itertool.chain() called q_net_params
    has both q networks'''
    q_net_optimizer = self.hparams.optim(q_net_params, lr=self.hparams.lr)
    policy_optimizer = self.hparams.optim(self.policy.parameters(), lr=self.hparams.lr)
    return [q_net_optimizer, policy_optimizer]

  def train_dataloader(self):
    dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
    dataloader = DataLoader(
        dataset= dataset,
        batch_size =self.hparams.batch_size,
    )
    return dataloader

  def training_step(self, batch, batch_idx, optimizer_idx):
    states, actions, rewards, dones, next_states = batch
    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)


    # Optimizer 1
    if optimizer_idx == 0:
      #Train Q-networks
      action_values1 = self.q_net1(states, actions)
      action_values2 = self.q_net2(states, actions)

      target_actions, target_log_probs = self.target_policy(next_states)

      next_action_values = torch.min(
          self.target_q_net1(next_states, target_actions),
          self.target_q_net2(next_states, target_actions)
      )
      next_action_values[dones] = 0.0

      expected_action_values = rewards + self.hparams.gamma * (next_action_values - self.hparams.alpha * target_log_probs)

      q_loss1 = self.hparams.loss_fn(action_values1, expected_action_values)
      q_loss2 = self.hparams.loss_fn(action_values2, expected_action_values)

      q_loss_total = q_loss1 + q_loss2
      self.log('epsiode/Q-Loss', q_loss_total)
      return q_loss_total


    # Optimizer 2
    elif optimizer_idx == 1:
      #Train Policy

      actions, log_probs = self.policy(states)

      action_values = torch.min(
          self.q_net1(states, actions),
          self.q_net2(states, actions)
      )

      policy_loss = (self.hparams.alpha * log_probs - action_values).mean()
      self.log("episode/ Policy Loss", policy_loss)


      return policy_loss

  def training_epoch_end(self, training_step_outputs):
    self.play_episode(policy=self.policy)

    
    polyak_average(self.q_net1, self.target_q_net1, tau=self.hparams.tau)
    polyak_average(self.q_net2, self.target_q_net2, tau=self.hparams.tau)

    polyak_average(self.policy, self.target_policy, tau=self.hparams.tau)

    self.log('episode/ episode_return', self.env.return_queue[-1])

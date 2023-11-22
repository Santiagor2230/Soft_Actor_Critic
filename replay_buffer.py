from collections import deque
import random

class ReplayBuffer:

  def __init__(self, capacity):
    self.buffer = deque(maxlen = capacity)

  def __len__(self):
    return len(self.buffer)

  def append(self, exp):
    self.buffer.append(exp)
  
  def sample(self, batch_size):
    return random.sample(self.buffer, batch_size)
import torch.nn as nn


class BaseDetector(nn.Module):
  """Head.
  """
  def __init__(self):
    super(BaseHead, self).__init__()

  def loss(self, x, target):
    """Calculate loss.
    """
    raise NotImplementedError()

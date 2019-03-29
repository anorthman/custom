import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ChannelShuffle(nn.Module):
  def __init__(self, group=1):
    assert group > 1
    super(ChannelShuffle, self).__init__()
    self.group = group
  def forward(self, x):
    """https://github.com/Randl/ShuffleNetV2-pytorch/blob/master/model.py
    """
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % self.group == 0)
    channels_per_group = num_channels // self.group
    # reshape
    x = x.view(batchsize, self.group, channels_per_group, height, width)
    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class SELayer(nn.Module):
  def __init__(self, channel, reduction=16):
    super(SELayer, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
        nn.Linear(channel, channel // reduction, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(channel // reduction, channel, bias=False),
        nn.Sigmoid())
  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    return x * y.expand_as(x)

# ResNet
# See https://github.com/aaron-xichen/pytorch-playground/blob/master/imagenet/resnet.py
def conv3x3(in_planes, out_planes, stride=1, groups=1):
  # "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, groups=groups,
                   stride=stride, padding=1, bias=False)

class MixedOp(nn.Module):
  """Mixed operation.
  Weighted sum of blocks.
  """
  def __init__(self, blocks):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for op in blocks:
      self._ops.append(op)

  def forward(self, x, weights):
    tmp = []
    for i, op in enumerate(self._ops):
      r = op(x)
      w = weights[..., i].reshape((-1, 1, 1, 1))
      res = w * r
      tmp.append(res)
    return sum(tmp)


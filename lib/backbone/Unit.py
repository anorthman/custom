import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .operator import SELayer, ChannelShuffle
from register import BASICUNIT
from collections import OrderedDict

# shufflenet_v2_baseunit
@BASICUNIT.register_module
class Shufv2Unit(nn.Module):
    def __init__(self, _in, _out, kernel_size=3, padding=1, stride=1, c_tag=0.5, activation=nn.ReLU, SE=False, residual=False, groups=2):
        super(Shufv2Unit, self).__init__()
        self.stride = stride
        self._in = _in
        self._out = _out
        self.groups = groups
        self.activation = activation(inplace=True)
        self.channel_shuffle = ChannelShuffle(group=groups)
        if self.stride == 1:
            assert _out == _in                    
            self.left_part = round(c_tag * _in)
            self.right_part_in = _in - self.left_part
            self.right_part_out = _out - self.left_part
            self.conv1 = nn.Conv2d(self.right_part_in, self.right_part_out, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.right_part_out)
            self.conv2 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=kernel_size, padding=padding, bias=False,
                                   groups=self.right_part_out)
            self.bn2 = nn.BatchNorm2d(self.right_part_out)
            self.conv3 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.right_part_out)
        elif self.stride == 2:
            assert _out == _in * 2
            self.conv1r = nn.Conv2d(_in, _in, kernel_size=1, bias=False)
            self.bn1r = nn.BatchNorm2d(_in)
            self.conv2r = nn.Conv2d(_in, _in, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=_in)
            self.bn2r = nn.BatchNorm2d(_in)
            self.conv3r = nn.Conv2d(_in, _in, kernel_size=1, bias=False)
            self.bn3r = nn.BatchNorm2d(_in)

            self.conv1l = nn.Conv2d(_in, _in, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=_in)
            self.bn1l = nn.BatchNorm2d(_in)
            self.conv2l = nn.Conv2d(_in, _in, kernel_size=1, bias=False)
            self.bn2l = nn.BatchNorm2d(_in)
        else:
            raise ValueError   

        # self.SE = SE
        # if self.SE:
        #     self.SELayer = SELayer(self.right_part_out, 2)  # TODO

    def forward(self, x):
        if self.stride == 1:
            left = x[:, :self.left_part, :, :]
            right = x[:, self.left_part:, :, :]
            out = self.conv1(right)
            out = self.bn1(out)
            out = self.activation(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out = self.activation(out)

            # if self.SE:
            #     out = self.SELayer(out)
            # if self.residual and self._in == self._out:
            #     out += right
            out = self.channel_shuffle(torch.cat((left, out), 1))
        elif self.stride == 2:
            out_r = self.conv1r(x)
            out_r = self.bn1r(out_r)
            out_r = self.activation(out_r)

            out_r = self.conv2r(out_r)
            out_r = self.bn2r(out_r)
            out_r = self.conv3r(out_r)
            out_r = self.bn3r(out_r)
            out_r = self.activation(out_r)
            #print(out_r)
            #print(out_r.size())
            #exit()

            out_l = self.conv1l(x)
            out_l = self.bn1l(out_l)
            out_l = self.conv2l(out_l)
            out_l = self.bn2l(out_l)
            out_l = self.activation(out_l)
            out = self.channel_shuffle(torch.cat((out_r, out_l), 1))
            #print(out)
            #print(out.size())
            #exit()
        else:
            raise ValueError            

        return out

# shufflent v2 downsample
class shufv2DownsampleUnit(nn.Module):
    def __init__(self, _in, c_tag=0.5, activation=nn.ReLU, groups=2):
        super(DownsampleUnit, self).__init__()

        self.conv1r = nn.Conv2d(_in, _in, kernel_size=1, bias=False)
        self.bn1r = nn.BatchNorm2d(_in)
        self.conv2r = nn.Conv2d(_in, _in, kernel_size=3, stride=2, padding=1, bias=False, groups=_in)
        self.bn2r = nn.BatchNorm2d(_in)
        self.conv3r = nn.Conv2d(_in, _in, kernel_size=1, bias=False)
        self.bn3r = nn.BatchNorm2d(_in)

        self.conv1l = nn.Conv2d(_in, _in, kernel_size=3, stride=2, padding=1, bias=False, groups=_in)
        self.bn1l = nn.BatchNorm2d(_in)
        self.conv2l = nn.Conv2d(_in, _in, kernel_size=1, bias=False)
        self.bn2l = nn.BatchNorm2d(_in)
        self.activation = activation(inplace=True)
        self.channel_shuffle = ChannelShuffle(group=groups)

        self.groups = groups
        self._in = _in

    def forward(self, x):
        out_r = self.conv1r(x)
        out_r = self.bn1r(out_r)
        out_r = self.activation(out_r)

        out_r = self.conv2r(out_r)
        out_r = self.bn2r(out_r)

        out_r = self.conv3r(out_r)
        out_r = self.bn3r(out_r)
        out_r = self.activation(out_r)

        out_l = self.conv1l(x)
        out_l = self.bn1l(out_l)

        out_l = self.conv2l(out_l)
        out_l = self.bn2l(out_l)
        out_l = self.activation(out_l)

        out = self.channel_shuffle(torch.cat((out_r, out_l), 1))

        return out

class pz_BasicUnit(nn.Module):
    def __init__(self, _in, _out, activation=nn.ReLU, SE=False, residual=False, groups=2):
        super(pz_BasicUnit, self).__init__()

        self.conv1 = nn.Conv2d(_in, _in, kernel_size=3, stride=1, padding=1, bias=False, groups=_in)
        self.bn1 = nn.BatchNorm2d(_in)
        self.conv2 = nn.Conv2d(_in,_in, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(_in)
        self.conv3 = nn.Conv2d(_in, _in, kernel_size=3, stride=1, padding=1, bias=False, groups=_in)
        self.bn3 = nn.BatchNorm2d(_in)
        self.conv4 = nn.Conv2d(_in,_out, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(_out)
        self.activation = activation(inplace=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.activation(out1)

        out2 = self.conv1(out1)
        out2 = self.bn1(out2)
        out2 = self.activation(out2)

        out3 = self.conv1(out2)
        out3 = self.bn1(out2)
        out3 = self.activation(out2)

        out4 = self.conv1(out3)
        out4 = self.bn1(out4)
        out4 = self.activation(out4)

        if self.SE:
            out4 = self.SELayer(out4)
        if self.residual and self._in == self._out:
            out4 += x
        return out4

class pz_DownsampleUnit(nn.Module):
    def __init__(self, _in, _out, activation=nn.ReLU, SE=False, residual=True, groups=2):
        super(pz_BasicUnit, self).__init__()

        self.conv1 = nn.Conv2d(_in, _in, kernel_size=3, stride=1, padding=1, bias=False, groups=_in)
        self.bn1 = nn.BatchNorm2d(_in)
        self.conv2 = nn.Conv2d(_in,_in, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(_in)
        self.conv3 = nn.Conv2d(_in, _in, kernel_size=3, stride=1, padding=1, bias=False, groups=_in)
        self.bn3 = nn.BatchNorm2d(_in)
        self.conv4 = nn.Conv2d(_in,_out, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(_out)
        self.activation = activation(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avepool = nn.Avepool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(x)
        out1 = self.activation(x)

        out2 = self.conv1(x)
        out2 = self.bn1(x)
        out2 = self.activation(x)

        out3 = self.conv1(x)
        out3 = self.bn1(x)
        out3 = self.activation(x)

        out4 = self.conv1(x)
        out4 = self.bn1(x)
        out4 = self.activation(x)

        out5 = self.maxpool(x)

        if self.SE:
            out4 = self.SELayer(out4)
        if self.residual:
            _out1 = self.avepool(x)
            _out1 = self.conv1(_out1)
            _out1 = self.bn1(_out1)
            _out1 = self.activation(_out1)

            _out2 = self.conv4(_out1)
            _out2 = self.bn4(_out2)
            _out3 = self.activation(_out2)
            out5 += _out3
        return out5


@BASICUNIT.register_module
class ResNetBasicBlock(nn.Module):
  def __init__(self, _in, _out, kernel_size=3, padding=1, stride=1, groups=1, shuffle=False,
               expansion=1):
    super(ResNetBasicBlock, self).__init__()
    m = OrderedDict()
    m['conv1'] = nn.Conv2d(_in, _out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    #m['conv1'] = conv3x3(_in, _out * expansion, stride) # groups=groups
    # if shuffle and groups > 1:
    #   m['shuffle1'] = ChannelShuffle(groups)
    m['bn1'] = nn.BatchNorm2d(_out)
    m['relu1'] = nn.ReLU(inplace=True)
    m['conv2'] = nn.Conv2d(_out, _out*expansion, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
    # if shuffle and groups > 1:
    #   m['shuffle2'] = ChannelShuffle(groups)
    m['bn2'] = nn.BatchNorm2d(_out*expansion)
    self.group1 = nn.Sequential(m)
    self.relu= nn.Sequential(nn.ReLU(inplace=True))
    downsample = None
    if stride != 1 or _in != _out:
      downsample = nn.Sequential(
          nn.Conv2d(_in, _out, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(_out))
    self.downsample = downsample
  def forward(self, x):
    if self.downsample is not None:
        residual = self.downsample(x)
    else:
        residual = x
    out = self.group1(x) + residual
    out = self.relu(out)
    return out

@BASICUNIT.register_module
class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()
  def forward(self, x):
    return x

@BASICUNIT.register_module
class ResNetBottleneck(nn.Module):
  def __init__(self, _in, _out, kernel_size=3, padding=1, stride=1, groups=1, shuffle=False, 
               expansion=1):
    super(ResNetBottleneck, self).__init__()
    m  = OrderedDict()
    r = int(_out * 0.25 * expansion)
    m['bn0'] = nn.BatchNorm2d(_in)
    m['relu0'] = nn.ReLU(inplace=True)
    m['conv1'] = nn.Conv2d(_in, r, kernel_size=1, bias=False)
    m['bn1'] = nn.BatchNorm2d(r)
    m['relu1'] = nn.ReLU(inplace=True)
    m['conv2'] = nn.Conv2d(r, r, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    m['bn2'] = nn.BatchNorm2d(r)
    m['relu2'] = nn.ReLU(inplace=True)
    m['conv3'] = nn.Conv2d(r, _out, groups=groups, kernel_size=1, bias=False)
    if shuffle and groups > 1:
      m['shuffle2'] = ChannelShuffle(groups)
    #m['bn3'] = nn.BatchNorm2d(_out)
    self.group1 = nn.Sequential(m)
    #self.relu= nn.Sequential(nn.ReLU(inplace=True))
    downsample = None
    if stride != 1 or _in != _out:
      downsample = nn.Sequential(
          nn.Conv2d(_in, _out, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(_out))
    self.downsample = downsample

  def forward(self, x):
    if self.downsample is not None:
        residual = self.downsample(x)
    else:
        residual = x
    out = self.group1(x) + residual
    #out = self.relu(out)
    return out

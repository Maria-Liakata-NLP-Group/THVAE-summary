# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from thirdparty.swish import Swish as SwishFN
from thirdparty.inplaced_sync_batchnorm import SyncBatchNormSwish

from utils import average_tensor
from collections import OrderedDict

BN_EPS = 1e-5
SYNC_BN = False
kernels = [1, 2, 3, 4, 5]


OPS = OrderedDict([
    ('res_elu', lambda Cin, Cout, stride: ELUConv(Cin, Cout, 3, stride, 1)),
    ('res_bnelu', lambda Cin, Cout, stride: BNELUConv(Cin, Cout, 3, stride, 1)),
    ('res_bnswish', lambda Cin, Cout, stride: BNSwishConv(Cin, Cout, 3, stride, 1)),
    ('res_bnswish5', lambda Cin, Cout, stride: BNSwishConv(Cin, Cout, 3, stride, 2, 2)),
    ('mconv_e6k5g0', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=6, dil=1, k=5, g=0)),
    ('mconv_e3k5g0', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=3, dil=1, k=5, g=0)),
    ('mconv_e3k5g8', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=3, dil=1, k=5, g=8)),
    ('mconv_e6k11g0', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=6, dil=1, k=11, g=0)),
])


def get_skip_connection(C, stride, affine, channel_mult):
    if stride == 1:
        # 不区分参数的占位符标识运算符。其实意思就是这个网络层的设计是用于占位的，即不干活，
        # 只是有这么一个层，放到残差网络里就是在跳过连接的地方用这个层，显得没有那么空虚！
        return Identity()
    elif stride == 2:
        return FactorizedReduce(C, int(channel_mult * C))
    elif stride == -1:
        # return nn.Sequential(UpSample(), Conv1D(C, int(C / channel_mult), kernel_size=1))
        return Conv1D(C, int(C / channel_mult), kernel_size=1)


def norm(t, dim):
    return torch.sqrt(torch.sum(t * t, dim))


def logit(t):
    return torch.log(t) - torch.log(1 - t)


def act(t):
    # The following implementation has lower memory.
    return SwishFN.apply(t)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return act(x)

@torch.jit.script
def normalize_weight_jit(log_weight_norm, weight):
    n = torch.exp(log_weight_norm)
    wn = torch.sqrt(torch.sum(weight * weight, dim=[1, 2]))   # norm(w)
    weight = n * weight / (wn.view(-1, 1, 1) + 1e-5)
    return weight


class Conv1D(nn.Conv1d):
    """Allows for weights as input.
    apply on text, should use 1D"""

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, data_init=False,
                 weight_norm=True):
        """
        Args:
            use_shared (bool): Use weights for this layer or not?
        """
        super(Conv1D, self).__init__(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias)

        self.log_weight_norm = None
        if weight_norm:
            # weight: [channel_out, channel_in / group, kernel_size]
            # 按照conv1d对weight进行处理， 这样channel_in/group =1, kernel is 1
            init = norm(self.weight, dim=[1, 2]).view(-1, 1, 1)
            self.log_weight_norm = nn.Parameter(torch.log(init + 1e-2), requires_grad=True)

        self.data_init = data_init
        self.init_done = False
        self.weight_normalized = self.normalize_weight()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, W).
            params (ConvParam): containing `weight` and `bias` (optional) of conv operation.
        """
        # do data based initialization
        if self.data_init and not self.init_done:
            with torch.no_grad():
                weight = self.weight / (norm(self.weight, dim=[1, 2]).view(-1, 1, 1) + 1e-5)
                bias = None
                out = F.conv1d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
                mn = torch.mean(out, dim=[0, 1, 2])
                st = 5 * torch.std(out, dim=[0, 1, 2])

                # get mn and st from other GPUs
                average_tensor(mn, is_distributed=True)
                average_tensor(st, is_distributed=True)

                if self.bias is not None:
                    self.bias.data = - mn / (st + 1e-5)
                self.log_weight_norm.data = -torch.log((st.view(-1, 1, 1) + 1e-5))
                self.init_done = True

        self.weight_normalized = self.normalize_weight()

        bias = self.bias

        return F.conv1d(x, self.weight_normalized, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def normalize_weight(self):
        """ applies weight normalization """
        if self.log_weight_norm is not None:
            weight = normalize_weight_jit(self.log_weight_norm, self.weight)
        else:
            weight = self.weight

        return weight


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SyncBatchNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SyncBatchNorm, self).__init__()
        self.bn = nn.SyncBatchNorm(*args, **kwargs)

    def forward(self, x):
        # Sync BN only works with distributed data parallel with 1 GPU per process. I don't use DDP, so I need to let
        # Sync BN to know that I have 1 gpu per process.
        self.bn.ddp_gpu_size = 1
        return self.bn(x)


# quick switch between multi-gpu, single-gpu batch norm
def get_batchnorm(*args, **kwargs):
    if SYNC_BN:
        return SyncBatchNorm(*args, **kwargs)
    else:
        # return nn.BatchNorm2d(*args, **kwargs)
        return nn.BatchNorm1d(*args, **kwargs)


class ELUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1):
        super(ELUConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.conv_0 = Conv1D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation,
                             data_init=True)

    def forward(self, x):
        out = F.elu(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        return out


class BNELUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1):
        super(BNELUConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.bn = get_batchnorm(C_in, eps=BN_EPS, momentum=0.05)
        self.conv_0 = Conv1D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation)

    def forward(self, x):
        x = self.bn(x)
        out = F.elu(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        return out


class BNSwishConv(nn.Module):
    """ReLU + Conv2d + BN."""

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1):
        super(BNSwishConv, self).__init__()
        # self.upsample = stride == -1 文本不用执行上采样或者下采样
        self.upsample = 0
        stride = abs(stride)
        self.bn = get_batchnorm(C_in, eps=BN_EPS, momentum=0.05)
        # self.bn_act = SyncBatchNormSwish(C_in, eps=BN_EPS, momentum=0.05)
        self.conv_0 = Conv1D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation)
        # self.conv_list = []
        # for kernel, feature in zip(kernel_size, C_out):
        #     self.conv_0 = Conv1D(C_in, feature, kernel, stride=stride, padding=padding, bias=True, dilation=dilation)
        #     self.conv_list.append(self.conv_0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W)
        """
        out = x
        # out = self.bn_act(x)
        # out = self.bn(x)
        out_list = []
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        # for conv in self.conv_list:
        #     s = conv(out)
        #     out_list.append(s)
        #     print(s.size(),'out ----------------')
        # out = torch.cat(out_list, dim=1)
        return out



class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        # self.conv_1 = Conv1D(C_in, C_out, 1, stride=2, padding=0, bias=True)
        self.conv_1 = Conv1D(C_in, C_out // 2, 2, stride=2, padding=0, bias=True)
        self.conv_2 = Conv1D(C_in, C_out - 1 * (C_out // 2), 3, stride=2, padding=1, bias=True)
        # self.conv_2 = Conv1D(C_in, C_out // 4, 1, stride=4, padding=0, bias=True)
        # self.conv_3 = Conv1D(C_in, C_out // 4, 1, stride=4, padding=0, bias=True)
        # self.conv_4 = Conv1D(C_in, C_out - 3 * (C_out // 4), 1, stride=4, padding=0, bias=True)

    def forward(self, x):
        out = act(x)
        conv1 = self.conv_1(out)
        # print(conv1.size(), 'conv1 -=-=-=-=-=-=-==')
        # adaptive = nn.AdaptiveMaxPool1d(conv1.size()[-1])
        # out = adaptive(out)
        conv2 = self.conv_2(out[:, :, 1:])
        # print(conv2.size(), 'cobv2 ---===========')
        # conv2 = adaptive(conv2)
        # conv3 = self.conv_3(out[:, :, 2:])
        # conv3 = adaptive(conv3)
        # conv4 = self.conv_4(out[:, :, 3:])
        # conv4 = adaptive(conv4)
        # out = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        # print(out.size(), 'ou 0-0-0-0-0-0-0-0-0-0-0')
        out = torch.cat([conv1, conv2], dim=1)

        return out


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        pass

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


class EncCombinerCell(nn.Module):
    def __init__(self, Cin1, Cin2, Cout, cell_type):
        super(EncCombinerCell, self).__init__()
        self.cell_type = cell_type
        # Cin = Cin1 + Cin2
        self.conv = nn.Conv1d(Cin2, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        x2 = self.conv(x2)
        out = x1 + x2
        return out


# original combiner
class DecCombinerCell(nn.Module):
    def __init__(self, Cin1, Cin2, Cout, cell_type):
        super(DecCombinerCell, self).__init__()
        self.cell_type = cell_type
        self.conv = Conv1D(Cin1 + Cin2, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)
        return out


class ConvBNSwish(nn.Module):
    def __init__(self, Cin, Cout, k=3, stride=1, groups=1, dilation=1):
        padding = dilation * (k - 1) // 2
        super(ConvBNSwish, self).__init__()

        self.conv = nn.Sequential(
            Conv1D(Cin, Cout, k, stride, padding, groups=groups, bias=False, dilation=dilation, weight_norm=False),
            # SyncBatchNormSwish(Cout, eps=BN_EPS, momentum=0.05)  # drop in replacement for BN + Swish
        )

    def forward(self, x):
        return self.conv(x)


class SE(nn.Module):
    def __init__(self, Cin, Cout):
        super(SE, self).__init__()
        num_hidden = max(Cout // 16, 4)
        self.se = nn.Sequential(nn.Linear(Cin, num_hidden), nn.ReLU(inplace=True),
                                nn.Linear(num_hidden, Cout), nn.Sigmoid())

    def forward(self, x):
        se = torch.mean(x, dim=[2])
        se = se.view(se.size(0), -1)
        se = self.se(se)
        se = se.view(se.size(0), -1, 1)
        return x * se


class InvertedResidual(nn.Module):
    def __init__(self, Cin, Cout, stride, ex, dil, k, g):
        # ex = 6, dil =1, k = 5, g=0
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2, -1]

        hidden_dim = int(round(Cin * ex))
        self.use_res_connect = self.stride == 1 and Cin == Cout
        # self.upsample = self.stride == -1
        self.upsample = 0
        self.stride = abs(self.stride)
        groups = hidden_dim if g == 0 else g


        layers0 = [nn.UpsamplingNearest2d(scale_factor=2)] if self.upsample else []
        layers = [get_batchnorm(Cin, eps=BN_EPS, momentum=0.05),
                  ConvBNSwish(Cin, hidden_dim, k=1),
                  ConvBNSwish(hidden_dim, hidden_dim, stride=self.stride, groups=groups, k=k, dilation=dil),
                  Conv1D(hidden_dim, Cout, 1, 1, 0, bias=False, weight_norm=False),
                  get_batchnorm(Cout, momentum=0.05)]

        layers0.extend(layers)
        self.conv = nn.Sequential(*layers0)

    def forward(self, x):
        return self.conv(x)

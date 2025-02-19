import math 
import torch 
import warnings
import torch.nn as nn 

from torch import Tensor
from mmpose.registry import MODELS
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from typing import Optional, Sequence, Tuple


def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Applies a convolution, batch normalization, and activation function to an input tensor in a neural network."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)  # 输入通道: c1, 输出通道：c_ , 卷积核：3x3, 步长1
        self.cv2 = Conv(c_, c2, k[1], 1, g=g) # 输入通道：c_ , 输出通道c2, 卷积核：3x3, 步长1
        self.add = shortcut and c1 == c2  # 当传入的shortcut参数为true，且c1和c2相等时，则使用残差连接。

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    """Implements a CSP Bottleneck module with three convolutions for enhanced feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    """Implements a fast Spatial Pyramid Pooling (SPPF) layer for efficient feature extraction in YOLOv5 models."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations for feature extraction."""
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    """Concatenates tensors along a specified dimension for efficient tensor manipulation in neural networks."""

    def __init__(self, dimension=1):
        """Initializes a Concat module to concatenate tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Concatenates a list of tensors along a specified dimension; `x` is a list of tensors, `dimension` is an
        int.
        """
        return torch.cat(x, self.d)


class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


@MODELS.register_module()
class Yolosv8BackboneChannelx2(BaseModule):
    def __init__(self, norm_eval: bool=False):
        super().__init__()

        # self.names = [str(i) for i in range(nc)]  # default names
        # BackBone
        self.conv1 = Conv(3, 64, 3, 2, 1)
        # self.conv1 = Conv(3, 32, 3, 2, 1)
        # self.conv1_stem = Conv(32, 32, 3, 1, 1)   # add for alux conv stem 
        self.conv2 = Conv(64, 128, 3, 2, 1)
        self.C1 = C2f(128, 128)
        self.conv3 = Conv(128, 256, 3, 2, 1)
        self.C2 = C2f(256, 256, n=2)
        self.conv4 = Conv(256, 512, 3, 2, 1)
        self.C3 = C2f(512, 512, n=3)
        self.conv5 = Conv(512, 1024, 3, 2)
        self.C4 = C2f(1024, 1024, n=1)
        self.spp1 = SPPF(1024, 1024, k=5)

        self.norm_eval = norm_eval
        
    def train(self, mode=True) -> None:
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        outs = []
        x = self.conv1(x)
        # x = self.conv1_stem(x)
        x = self.conv2(x)
        x = self.C1(x)
        x = self.conv3(x)
        c2 = self.C2(x)

        x = self.conv4(c2)
        c3 = self.C3(x)
        x = self.conv5(c3)  
        # print(x)
        x = self.C4(x)
        x = self.spp1(x)

        # get last feature
        outs.append(x)
        return tuple(outs)
    

if __name__ == "__main__":
    inputs = torch.randn(1, 3, 256, 192).float()
    model = Yolosv8BackboneChannelx2()

    outputs = model(inputs)
    print(outputs[0].shape)
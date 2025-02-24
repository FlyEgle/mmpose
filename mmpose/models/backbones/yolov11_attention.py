import torch
import warnings
import torch.nn as nn

from torch import Tensor
from mmpose.registry import MODELS
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from typing import Optional, Sequence, Tuple


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None):  # kernel, padding
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3K2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


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


@MODELS.register_module()
class Yolosv11BackboneChannelx2AllStageAttention(BaseModule):
    def __init__(self, norm_eval: bool=False):
        super().__init__()

        # self.names = [str(i) for i in range(nc)]  # default names
        # BackBone
        self.conv1 = Conv(3, 64, 3, 2, 1)
        # self.conv1 = Conv(3, 32, 3, 2, 1)
        # self.conv1_stem = Conv(32, 32, 3, 1, 1)   # add for alux conv stem 
        self.conv2 = Conv(64, 128, 3, 2, 1)
        self.C1 = C3K2(128, 128, shortcut=False)
        self.c2psa1 = C2PSA(128, 128)
        self.conv3 = Conv(128, 256, 3, 2, 1)
        self.C2 = C3K2(256, 256, n=2)
        self.c2psa2 = C2PSA(256, 256)
        self.conv4 = Conv(256, 512, 3, 2, 1)
        self.C3 = C3K2(512, 512, n=3)
        self.c2psa3 = C2PSA(512, 512)
        self.conv5 = Conv(512, 1024, 3, 2)
        self.C4 = C3K2(1024, 1024, n=1)
        self.spp1 = SPPF(1024, 1024, k=5)
        self.c2psa4 = C2PSA(1024, 1024)

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
        x = self.c2psa1(x)
        x = self.conv3(x)
        c2 = self.C2(x)
        c2 = self.c2psa2(x)

        x = self.conv4(c2)
        c3 = self.C3(x)
        c3 = self.c2psa3(c3)
        x = self.conv5(c3)  
        # print(x)
        x = self.C4(x)
        x = self.spp1(x)
        x = self.c2psa4(x)

        # get last feature
        outs.append(x)
        return tuple(outs)


@MODELS.register_module()
class Yolosv11BackboneChannelx2ThreeStageAttention(BaseModule):
    def __init__(self, norm_eval: bool=False):
        super().__init__()

        # self.names = [str(i) for i in range(nc)]  # default names
        # BackBone
        self.conv1 = Conv(3, 64, 3, 2, 1)
        # self.conv1 = Conv(3, 32, 3, 2, 1)
        # self.conv1_stem = Conv(32, 32, 3, 1, 1)   # add for alux conv stem 
        self.conv2 = Conv(64, 128, 3, 2, 1)
        self.C1 = C3K2(128, 128, shortcut=False)
        # self.c2psa1 = C2PSA(128, 128)
        self.conv3 = Conv(128, 256, 3, 2, 1)
        self.C2 = C3K2(256, 256, n=2)
        self.c2psa2 = C2PSA(256, 256)
        self.conv4 = Conv(256, 512, 3, 2, 1)
        self.C3 = C3K2(512, 512, n=3)
        self.c2psa3 = C2PSA(512, 512)
        self.conv5 = Conv(512, 1024, 3, 2)
        self.C4 = C3K2(1024, 1024, n=1)
        self.spp1 = SPPF(1024, 1024, k=5)
        self.c2psa4 = C2PSA(1024, 1024)

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
        # x = self.c2psa1(x)
        x = self.conv3(x)
        c2 = self.C2(x)
        c2 = self.c2psa2(x)

        x = self.conv4(c2)
        c3 = self.C3(x)
        c3 = self.c2psa3(c3)
        x = self.conv5(c3)  
        # print(x)
        x = self.C4(x)
        x = self.spp1(x)
        x = self.c2psa4(x)

        # get last feature
        outs.append(x)
        return tuple(outs)


@MODELS.register_module()
class Yolosv11BackboneChannelx2TwoStageAttention(BaseModule):
    def __init__(self, norm_eval: bool=False):
        super().__init__()

        # self.names = [str(i) for i in range(nc)]  # default names
        # BackBone
        self.conv1 = Conv(3, 64, 3, 2, 1)
        # self.conv1 = Conv(3, 32, 3, 2, 1)
        # self.conv1_stem = Conv(32, 32, 3, 1, 1)   # add for alux conv stem 
        self.conv2 = Conv(64, 128, 3, 2, 1)
        self.C1 = C3K2(128, 128, shortcut=False)
        # self.c2psa1 = C2PSA(128, 128)
        self.conv3 = Conv(128, 256, 3, 2, 1)
        self.C2 = C3K2(256, 256, n=2)
        # self.c2psa2 = C2PSA(256, 256)
        self.conv4 = Conv(256, 512, 3, 2, 1)
        self.C3 = C3K2(512, 512, n=3)
        self.c2psa3 = C2PSA(512, 512)
        self.conv5 = Conv(512, 1024, 3, 2)
        self.C4 = C3K2(1024, 1024, n=1)
        self.spp1 = SPPF(1024, 1024, k=5)
        self.c2psa4 = C2PSA(1024, 1024)

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
        # x = self.c2psa1(x)
        x = self.conv3(x)
        c2 = self.C2(x)
        # c2 = self.c2psa2(x)

        x = self.conv4(c2)
        c3 = self.C3(x)
        c3 = self.c2psa3(c3)
        x = self.conv5(c3)  
        # print(x)
        x = self.C4(x)
        x = self.spp1(x)
        x = self.c2psa4(x)

        # get last feature
        outs.append(x)
        return tuple(outs)


# 测试代码
if __name__ == "__main__":
    inputs = torch.randn(1, 3, 256, 192).float()
    model = Yolosv11BackboneChannelx2TwoStageAttention()

    outputs = model(inputs)
    print(outputs[0].shape)
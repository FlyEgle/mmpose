import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from models.common import *
from utils import torch_utils
from utils.general import check_version, check_yaml, make_divisible, print_args, LOGGER
from utils.autoanchor import check_anchor_order
from models.experimental import attempt_load

from mmpose.models import BACKBONES



class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=None):  # detection layer
        super(Detect, self).__init__()
        if anchors is None:
            Anchors = [
                [10,13, 16,30, 33,23],     # P3/8
                [30,61, 62,45, 59,119],    # P4/16
                [116,90, 156,198, 373,326] # P5/32
            ]
        else:
            Anchors = anchors
        ch = [128, 256, 512]
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(Anchors)  # number of detection layers
        self.na = len(Anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.stride = torch.tensor([8, 16, 32]).float()
        self.register_buffer('anchors', torch.tensor(Anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)  # each anchor group match a result
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid



class YoloSModel(nn.Module):
    def __init__(self, nc=80, anchors=None):
        super(YoloSModel, self).__init__()

        self.names = [str(i) for i in range(nc)]  # default names
        # BackBone
        self.conv1 = Conv(3, 32, 6, 2, 2)
        # self.conv1 = Conv(3, 32, 3, 2, 1)
        self.conv1_stem = Conv(32, 32, 3, 1, 1)   # add for alux conv stem 
        self.conv2 = Conv(32, 64, 3, 2, 1)
        self.C1 = C3(64, 64)
        self.conv3 = Conv(64, 128, 3, 2, 1)
        self.C2 = C3(128, 128, n=2)
        self.conv4 = Conv(128, 256, 3, 2, 1)
        self.C3 = C3(256, 256, n=3)
        self.conv5 = Conv(256, 512, 3, 2)
        self.C4 = C3(512, 512, n=1)
        self.spp1 = SPPF(512, 512, k=5)
        
        # Neck
        self.conv6 = Conv(512, 256, 1, 1)
        self.up1 = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.cat1 = Concat()
        self.C5 = C3(512, 256, shortcut=False)
        
        self.conv7 = Conv(256, 128, 1, 1)
        self.up2 = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.cat2 = Concat()
        self.C6 = C3(256, 128, shortcut=False)

        self.conv8 = Conv(128, 128, 3, 2)
        self.cat3 = Concat()
        self.C7 = C3(256, 256, shortcut=False)

        self.conv9 = Conv(256, 256, 3, 2)
        self.cat4 = Concat()
        self.C8 = C3(512, 512, shortcut=False)

        self.head = Detect(nc, anchors)
        self.stride = self.head.stride

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv1_stem(x)
        x = self.conv2(x)
        x = self.C1(x)
        x = self.conv3(x)
        c2 = self.C2(x)

        x = self.conv4(c2)
        c3 = self.C3(x)
        # print(c3.shape, c2.shape)
        x = self.conv5(c3)  
        # print(x)
        x = self.C4(x)
        x = self.spp1(x)
        x1 = self.conv6(x)

        x1_up = self.up1(x1)
        c3_x1 = self.cat1([x1_up, c3])
        # print(x1_up, c3)
        
        # ==================
        x = self.C5(c3_x1)
        x2 = self.conv7(x)

        x2_up = self.up2(x2)
        # ==================
        c2_x2 = self.cat2([x2_up, c2])
        # print(x2_up, c2)

        # head1 
        p1 = self.C6(c2_x2)   # (n, 128, 80, 80)

        p1_c = self.conv8(p1)
        p1_x2 = self.cat3([p1_c, x2])
        # print(p1_c, x2)
        # head2 
        p2 = self.C7(p1_x2)  # (n, 256, 40, 40)
        
        p2_c = self.conv9(p2)
        p2_x1 = self.cat4([p2_c, x1])
        # print(p2_c, x1)
        
        # head 3
        p3 = self.C8(p2_x1)  # (n, 512, 20, 20)

        pred = self.head([p1, p2, p3])
        return pred
    
    
@BACKBONES.register_module()
class YOLOBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super(YOLOBackbone, self).__init__()
        self.pretrained = pretrained
        
        pass
    
        
        
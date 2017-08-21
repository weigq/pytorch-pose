'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import math
# from .preresnet import BasicBlock, BottleNeck

__all__ = ['HourglassNet2', 'hg21', 'hg22', 'hg24', 'hg28']

class BottleNeck2(nn.Module):
    expansion = 2

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BottleNeck2, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes/2, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes/2)
        self.conv2 = nn.Conv2d(out_planes/2, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # element-wise add
        out += residual

        return out

class Hourglass2(nn.Module):
    def __init__(self, block, planes, depth=4, num_blocks=1):
        super(Hourglass2, self).__init__()
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hourglass(block, planes, depth, num_blocks)

    def _make_residual(self, block, planes, num_blocks=1):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def _make_hourglass(self, block, planes, depth=4, num_blocks=1):
        hg = []
        for i in range(depth):
            res = []
            for j in range(depth - 1): 
                res.append(self._make_residual(block, planes, num_blocks))
            if i == 0:
                res.append(self._make_residual(block, planes, num_blocks))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)


    def _hourglass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hourglass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hourglass_forward(self.depth, x)


class HourglassNet2(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=4, num_blocks=1, num_classes=16):
        super(HourglassNet2, self).__init__()

        self.inplanes = 128
        self.num_feats = 128
        self.num_stacks = num_stacks


        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.res1 = self._make_residual(block, 64, 128)
        self.res2 = self._make_residual(block, 128, 128)
        self.res3 = self._make_residual(block, 128, 256)

        

        # build hourglass modules
        # ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []

        for i in range(self.num_stacks):
            hg.append(Hourglass2(block, 256, 4, num_blocks))
            res.append(self._make_residual(block, 256, 256, num_blocks))
            fc.append(self._make_fc(256, 256))
            score.append(nn.Conv2d(256, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(256, 256, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, 256, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, in_planes, out_planes, blocks=1, stride=1):
        '''a residual module'''
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True))

        layers = []
        layers.append(block(in_planes, out_planes, stride, downsample))

        for i in range(1, blocks):
            layers.append(block(out_planes, out_planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.res1(x)
        x = self.maxpool(x)
        x = self.res2(x)
        x = self.res3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out

def hg21(**kwargs):
    model = HourglassNet2(BottleNeck2, num_stacks=1, num_blocks=1, **kwargs)
    return model

def hg22(**kwargs):
    model = HourglassNet2(BottleNeck2, num_stacks=2, num_blocks=1, **kwargs)
    return model

def hg24(**kwargs):
    model = HourglassNet2(BottleNeck2, num_stacks=4, num_blocks=1, **kwargs)
    return model

def hg28(**kwargs):
    model = HourglassNet2(BottleNeck2, num_stacks=8, num_blocks=1, **kwargs)
    return model

'''
the hourglass model used for multi-person pose estimation 
'''
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['HourglassNet', 'hg1', 'hg2', 'hg4', 'hg8']

class Hourglass(nn.Module):
    def __init__(self, planes, depth=4, num_blocks=1):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hourglass(planes, depth, num_blocks)

    def _make_residual(self, in_planes, out_planes, num_blocks=1):
        '''
        the stacks of conv3x3 with number of num_blocks
        '''
        layers = []
        for i in range(0, num_blocks):
            layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True))
        return nn.Sequential(*layers)

    def _make_hourglass(self, planes, depth=4, num_blocks=1):
        hg = []
        hg_planes = [768, 512, 386, 256]
        for i in range(self.depth):
            res = []
            # up1 branch
            res.append(self._make_residual(hg_planes[i], hg_planes[i], num_blocks))
            # low branch
            if i == 0:
                res.append(self._make_residual(hg_planes[i], hg_planes[i], num_blocks))
                res.append(self._make_residual(hg_planes[i], hg_planes[i], num_blocks))
                res.append(self._make_residual(hg_planes[i], hg_planes[i], num_blocks))
            else:
                res.append(self._make_residual(hg_planes[i], hg_planes[i-1], num_blocks))
                res.append(self._make_residual(hg_planes[i-1], hg_planes[i], num_blocks))
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


class HourglassNet(nn.Module):
    '''create the hourglass model '''
    def __init__(self, num_stacks=4, num_blocks=1, num_classes=16):
        '''
        params:
        block:        the block used in the residual module
        num_stacks:   the number of the hg stages
        num_blocks:   the number of blocks in each residual module
        num_classes:  the number of joints classes 
        '''
        super(HourglassNet, self).__init__()

        self.num_stacks = num_stacks

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.res1 = self._make_residual(64, 128)
        self.res2 = self._make_residual(128, 128)
        self.res3 = self._make_residual(128, 256)

        # build hourglass modules
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(self.num_stacks):
            hg.append(Hourglass(planes=256, depth=4, num_blocks=num_blocks))
            res.append(self._make_residual(256, 256))
            fc.append(self._make_fc(256, 256))
            score.append(nn.Conv2d(256, num_classes*2, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(256, 256, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes*2, 256, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, in_planes, out_planes, num_blocks=1, stride=1):
        '''a residual module, whose input planes are in_planes, and outputs out_planes
           the resolution will not be changed
        '''
        layers = []
        for i in range(0, num_blocks):
            layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True))

        return nn.Sequential(*layers)

    def _make_fc(self, in_planes, out_planes):
        bn = nn.BatchNorm2d(out_planes)
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=True)
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

# ==========================================
# 4 types HG models with different parameters
# ==========================================

def hg1(**kwargs):
    model = HourglassNet(num_stacks=1, **kwargs)
    return model

def hg2(**kwargs):
    model = HourglassNet(num_stacks=2, **kwargs)
    return model

def hg4(**kwargs):
    model = HourglassNet(num_stacks=4, **kwargs)
    return model

def hg8(**kwargs):
    model = HourglassNet(num_stacks=8, **kwargs)
    return model

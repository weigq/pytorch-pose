'''
load dataset
'''

from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math
import scipy.misc

import torch
import torch.utils.data as data

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *

class Mpii(data.Dataset):
    '''
        the subclass of data.Dataset
        should override __len__ and __getitem__
    '''
    def __init__(self, jsonfile, img_path, inp_res=256, out_res=64, train=True, sigma=1, scale_factor=0.25, rot_factor=30):
        self.img_path = img_path
        self.is_train = train           # training set or test set
        '''
        the input and output resolution should be changed
        '''
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor

        # create train/val split
        with open(jsonfile) as anno_file:
            self.anno = json.load(anno_file)

        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            if val['isValidation'] == True:
                self.valid.append(idx)
            else:
                self.train.append(idx)
        self.train = self.train[0:4]
        self.valid = self.valid[0:1]

        self.mean, self.std = self._get_param()

    def _get_param(self):
        paramFile = './data/mpii/mean.pth.tar'
        if isfile(paramFile):
            param = torch.load(paramFile)
            #i = 1
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            print("len of train: {}".format(len(self.train)))
            i = 1
            for index in self.train:
                i += 1
                if i%500 == 0:
                    print("i = {}\n".format(i-1))
                ann = self.anno[index]
                img_path = os.path.join(self.img_path, ann['img_paths'])
                img = load_image(img_path) # CxHxW

                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train)
            std /= len(self.train)
            param = {
                'mean': mean,
                'std': std,
                }
            torch.save(param, paramFile)
        #if self.is_train:
        print('    Mean: %.4f, %.4f, %.4f' % (param['mean'][0], param['mean'][1], param['mean'][2]))
        print('    Std:  %.4f, %.4f, %.4f' % (param['std'][0], param['std'][1], param['std'][2]))
        return param['mean'], param['std']


    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]

        img_path = os.path.join(self.img_path, a['img_paths'])

        pts = torch.Tensor(a['joint_self'])
        # pts[:,0:2] -= 1 # Convert pts to zero based

        c = torch.Tensor(a['objpos'])
        s = a['scale_provided']

        # Adjust center/scale slightly to avoid cropping limbs
        if c[0] != -1:
            c[1] = c[1] + 15 * s
            s = s * 1.25

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)  # = 16
        #img = load_image(img_path) # CxHxW
        img = scipy.misc.imread(img_path)
        img = np.transpose(img, (2, 0, 1)) # C*H*W
        img = torch.from_numpy(img).float()
        # normalization
        if img.max() > 1:
            img /= 255
        

        r = 0
        if self.is_train:
            s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf,1+sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2*rf,2*rf)[0] if random.random() <= 0.6 else 0

            # Flip
            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='mpii')
                c[0] = img.size(2) - c[0]


            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)

        #f self.is_train:
        inp = color_normalize(inp, self.mean, self.std)

        # Generate ground truth
        tpts = pts.clone()

        # the output of the net
        target = torch.zeros(nparts, self.out_res, self.out_res)

        for i in range(nparts):
            if tpts[i, 0] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2] + 1, c, s, [self.out_res, self.out_res], rot=r))
                target[i] = draw_gaussian(target[i], tpts[i] - 1, self.sigma)

        # Meta info
        meta = {'index' : index,
                'center' : c,
                'scale' : s,
                'pts' : pts,
                'tpts' : tpts}

        # if self.is_train:

        #     # Color
        #     inp[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        #     inp[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        #     inp[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)


        #     return inp, target
        # else:
        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)

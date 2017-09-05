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
from torchvision import transforms

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *
#from pose.utils.imtransform import *


class Mpii(data.Dataset):
    '''
        the subclass of data.Dataset
        should override __len__ and __getitem__
    '''
    def __init__(self, jsonfile, img_path, inp_res=512, out_res=128, is_train=True, sigma=1, scale_factor=1.0, rot_factor=30):
        self.img_path = img_path
        self.is_train = is_train
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor

        self.transform = transforms.Compose([transforms.ToTensor()])

        # create train/val split
        with open(jsonfile) as anno_file:
            self.anno = json.load(anno_file)

        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            if val['isVal'] == 1:
                self.valid.append(idx)
            else:
                self.train.append(idx)
        # self.train = self.train[0:4]
        # self.valid = self.valid[0:1]

        if self.is_train == True:
            print("length of training set: {}".format(len(self.train)))
        else:
            print("length of validation set: {}".format(len(self.valid)))

        self.mean, self.std = self._get_param()


    def _get_param(self):
        param_file = './data/mpii/mean.pth.tar'
        if isfile(param_file):
            param = torch.load(param_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            i = 1
            for index in self.train:
                if i%1000 == 0:
                    print("processed {}/{} images\n".format(i, len(self.train)))
                i += 1

                ann = self.anno[index]
                img_path = os.path.join(self.img_path, ann['path'])
                img = load_img(img_path, self.transform) # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train)
            std /= len(self.train)
            param = {
                'mean': mean,
                'std': std,
                }
            torch.save(param, param_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (param['mean'][0], param['mean'][1], param['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (param['std'][0], param['std'][1], param['std'][2]))
        #raw_input(">>>")
        return param['mean'], param['std']


    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
        if self.is_train:
            ann = self.anno[self.train[index]]
        else:
            ann = self.anno[self.valid[index]]

        img_path = os.path.join(self.img_path, ann['path'])
        pts = np.array(ann['joints'])

        # print("{}>>>{}".format(pts.shape, ann['path']))
        # pts = []
        # for i in range(13):
        #     pts.append(ann['joints'][i])
        # # pts = torch.Tensor(pts)
        # print(":{}>{}".format(type(pts[0][0][0]), pts.shape))
        # raw_input("??")
        # print(">>{}".format(type(ann['joints'])))
        
        # c = torch.Tensor(a['objpos'])
        # s = a['scale_provided']
        # s = 1.0

        # Adjust center/scale slightly to avoid cropping limbs
        # if c[0] != -1:
        #     c[1] = c[1] + 15 * s
        #     s = s * 1.25

        # For single-person pose estimation with a centered/scaled figure

        npeople = ann['numPeople']  # number of people in single image
        # print(">>.{}".format(type(npeople)))
        nparts = pts.shape[1]  # = 16
        img = load_img(img_path, self.transform) # CxHxW

        # r = 0
        if self.is_train:
            # s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf,1+sf)[0]
            # r = torch.randn(1).mul_(rf).clamp(-2*rf,2*rf)[0] if random.random() <= 0.5 else 0

            # Flip
            # if random.random() <= 0.5:
            #     img = torch.from_numpy(fliplr(img.numpy())).float()
            #     pts = shufflelr(pts, width=img.size(2), dataset='mpii')
            #     c[0] = img.size(2) - c[0]


            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        s = img.size(1)
        c = list([img.size(2)/2, img.size(1)/2])

        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res])
        # print(".{}".format(inp.size()))
        # raw_input("??")
        # inp = crop_inp(img, self.inp_res, rot=r)

        #f self.is_train:
        inp = color_normalize(inp, self.mean, self.std)

        # Generate ground truth
        tpts = pts# .clone()

        # the output of the net
        target = torch.zeros(nparts, self.out_res, self.out_res)

        for n in range(npeople):
            for i in range(nparts):
                if tpts[n, i, 0] > 0:
                    trans = transform(tpts[n, i, 0:2] + 1, c, s, [self.out_res, self.out_res])
                    # print("{}".format(trans))
                    tpts[n, i, 0:2] = trans
                    target[i] = draw_gaussian(target[i], tpts[n, i] - 1, self.sigma)

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

from __future__ import absolute_import

import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import torch

from .misc import *
from .imutils import *


def crop_inp(img_orig, res = 512, center=0, scale=1, rot=0):
    '''
    imgIn
    type: ndarray
    shape: C*H*W
    '''
    # transpose img_orig to H*W*C
    img = img_orig.cpu().numpy()
    img = np.transpose(img, (1,2,0))
    h_orig, w_orig = img.shape[0], img.shape[1]
    if h_orig >= w_orig:
    cen = res / 2
    new_img = np.zeros((res, res, 3))
    new_img[cen - h_orig/2 : cen + h_orig - h_orig/2, cen - w_orig/2 : cen +w_orig - w_orig/2] = img[0 : h_orig, 0 : w_orig]

    new_img = np.transpose(new_img, (2,0,1))
    new_img = torch.from_numpy(new_img).float()
    print("max:.{}".format(new_img.max()))
    raw_input("???")

    # Upper left point
    # ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # # Bottom right point
    # br = np.array(transform(res, center, scale, res, invert=1))

    # # Padding so that when rotated proper amount of context is included
    # pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    # if not rot == 0:
    #     ul -= pad
    #     br += pad

    # new_shape = [br[1] - ul[1], br[0] - ul[0]]
    # if len(img.shape) > 2:
    #     new_shape += [img.shape[2]]
    # new_img = np.zeros(new_shape)

    # # Range to fill new array
    # new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    # new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # # Range to sample from original image
    # old_x = max(0, ul[0]), min(len(img[0]), br[0])
    # old_y = max(0, ul[1]), min(len(img), br[1])
    # new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    # if not rot == 0:
    #     # Remove padding
    #     new_img = scipy.misc.imrotate(new_img, rot)
    #     new_img = new_img[pad:-pad, pad:-pad]

    # new_img = im_to_torch(scipy.misc.imresize(new_img, res))
    return new_img

from __future__ import absolute_import

import os
import errno


__all__ = ['LRDecay', 'AverageMeter', 'mkdir']


def LRDecay(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by 0.2 every 20 epochs"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


#==========================================#
#some operations of file 
#==========================================#
def mkdir(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
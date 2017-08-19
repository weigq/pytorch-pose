from __future__ import absolute_import

import os
import errno



def isfile(fname):
    return os.path.isfile(fname) 

def isdir(dirname):
    return os.path.isdir(dirname)

def join(path, *paths):
    return os.path.join(path, *paths)

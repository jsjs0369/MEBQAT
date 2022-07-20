import functools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

"""
Written referring to:
    https://stackoverflow.com/a/31174427/6937913
    https://github.com/gngdb/pytorch-minimize
    https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
"""

def create_log_func(path):
    f = open(path, 'a')
    counter = [0]
    def log(txt):
        print(txt)
        f.write(txt + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
    return log, f.close

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def rdelattr(obj, attr):
    pre, _, post = attr.rpartition('.')
    delattr(rgetattr(obj, pre) if pre else obj, post)  

def rsetparam(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    (rgetattr(obj, pre) if pre else obj).register_parameter(post, val)

def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2.0) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

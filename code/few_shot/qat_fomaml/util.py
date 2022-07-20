import functools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

# Written referring to ...
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

## Written referring to https://stackoverflow.com/a/31174427/6937913 and https://github.com/gngdb/pytorch-minimize/blob/529aef1e47739442ccc6815f74f2a0090e21432b/pytorch_minimize/optim.py
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
##

## Written referring to https://tutorials.pytorch.kr/intermediate/tensorboard_tutorial.html
def plot_example(imgs, labels, N, K):
    imgs_cpu, labels_cpu = imgs.cpu(), labels.cpu()

    fig, axes = plt.subplots(K, N, figsize=(12, 48))
    
    for n in np.arange(N):
        for k in np.arange(K):
            idx = K * n + k
            img, label = imgs_cpu[idx], labels_cpu[idx]

            img = img / 2.0 + 0.5
            img = np.clip(img.numpy(), 0, 1)
            img = np.transpose(img, (1, 2, 0))

            if (N == 1) or (K == 1):
                axes[idx].imshow(img)
                axes[idx].set_title(f'({label})')
            else:
                axes[k, n].imshow(img)
                axes[k, n].set_title(f'({label})')                

    fig.tight_layout()

    return fig

def plot_prediction(imgs, preds, labels, N, K):
    imgs_cpu, preds_cpu, labels_cpu = imgs.cpu(), preds.cpu(), labels.cpu()
    preds_squeezed = np.squeeze(preds_cpu.numpy())

    fig, axes = plt.subplots(K, N, figsize=(12, 48))
    
    for n in np.arange(N):
        for k in np.arange(K):
            idx = K * n + k
            img, pred, label = imgs_cpu[idx], preds_squeezed[idx], labels_cpu[idx]                        

            img = img / 2.0 + 0.5
            img = np.clip(img.numpy(), 0, 1)
            img = np.transpose(img, (1, 2, 0))

            if (N == 1) or (K == 1):
                axes[idx].imshow(img)
                axes[idx].set_title(
                    f'{pred}\n({label})',
                    color=("green" if pred == label.item() else "red"))
            else:
                axes[k, n].imshow(img)
                axes[k, n].set_title(
                    f'{pred}\n({label})',
                    color=("green" if pred == label.item() else "red"))

    fig.tight_layout()

    return fig
##

# Written referring to https://github.com/pytorch/vision/blob/981ccfdff1e173d37712dd13d960a2495287aee0/torchvision/models/mobilenetv2.py#L16
def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2.0) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

# LSQ
class LSQ(autograd.Function):
    # Boot weight/activation quantization
    @staticmethod
    def boot(qb, x, target, prev_act=None, p=None):
        n, scale_grad, scale = None, None, None

        if target == 'w':   # Weight quantization
            # Config 2 (Signed, Symmetric)
            n = (-1.0) * (2.0 ** (qb - 1.0))
            p = (2.0 ** (qb - 1.0)) - 1.0
            scale_grad = 1.0 / math.sqrt(x.numel() * p)
            avg_abs = torch.mean(torch.abs(x)) 
            scale = nn.Parameter(2.0 * avg_abs / math.sqrt(p))
        else:               # Activation quantization
            assert target == 'a'

            if prev_act is not None:
                # Config 2 or 3 (Symmetric)
                if prev_act in ['relu', 'relu6']:
                    n = 0
                    p = (2.0 ** qb)
                elif prev_act == 'swish':
                    n = -1.0
                    p = (2.0 ** qb) - 1.0
                else:
                    assert prev_act == 'etc'
                    n = (-1.0) * (2.0 ** (qb - 1.0))
                    p = (2.0 ** (qb - 1.0)) - 1.0
                scale_grad = None
                scale = nn.Parameter(torch.Tensor())
            else:
                assert p is not None
                scale_grad = 1.0 / math.sqrt(x.numel() * p)         
                avg_abs = torch.mean(torch.abs(x))            
                scale = nn.Parameter(2.0 * avg_abs / math.sqrt(p))

        return n, p, scale_grad, scale

    # 'Fake' weight/activation quantization
    @staticmethod
    def forward(ctx, x, scale, scale_grad, n, p):
        ctx.save_for_backward(x, scale)
        ctx.others = scale_grad, n, p

        x_sc = x / (scale + 1e-6)
        x_cl = F.hardtanh(x_sc, min_val=n, max_val=p)
        x_bar = torch.round(x_cl)
        x_hat = x_bar * scale    

        return x_hat

    # Backpropagation in QAT
    @staticmethod
    def backward(ctx, dx_hat):
        x, scale = ctx.saved_tensors
        scale_grad, n, p = ctx.others

        x_sc = x / (scale + 1e-6)

        indicate_small = (x_sc < n).float()
        indicate_big = (x_sc > p).float()
        indicate_mid = 1.0 - indicate_small - indicate_big

        dx = dx_hat * indicate_mid

        dscale = (
            (indicate_small * n + indicate_big * p + indicate_mid * (torch.round(x_sc) - x_sc)) * dx_hat * scale_grad).sum().unsqueeze(dim=0)

        return dx, dscale, None, None, None

    # 'Practical' weight/activation quantization
    @staticmethod
    def quant(x, scale, n, p):
        x_sc = x / (scale + 1e-6)
        x_cl = F.hardtanh(x_sc, min_val=n, max_val=p)
        x_bar = torch.round(x_cl)

        return x_bar

## DoReFa-Net
class DoReFaSTEForW(autograd.Function):    # quantize_k in DoReFaNet paper
    @staticmethod
    def forward(ctx, x, p):
        # x, y: r_i, r_o in DoReFaNet paper, respectively
        # p = (2.0 ** qb) - 1.0
        if p == 1:  # if qb == 1
            y = torch.sign(x) * torch.mean(torch.abs(x))
        else:
            assert torch.all(x >= 0) and torch.all(x <= 1)
            y = torch.round(p * x) / p
            assert torch.all(y >= 0) and torch.all(y <= 1)

        return y

    @staticmethod
    def backward(ctx, dy):
        dx = dy.clone()

        return dx, None     

class DoReFaSTEForA(autograd.Function):    # quantize_k in DoReFaNet paper
    @staticmethod
    def forward(ctx, x, p):
        # x, y: r_i, r_o in DoReFaNet paper, respectively
        # p = (2.0 ** qb) - 1.0
        assert torch.all(x >= 0) and torch.all(x <= 1)
        y = torch.round(p * x) / p
        assert torch.all(y >= 0) and torch.all(y <= 1)

        return y

    @staticmethod
    def backward(ctx, dy):
        dx = dy.clone()

        return dx, None 

class DoReFa:
    @staticmethod
    def boot(qb):
        p = (2.0 ** qb) - 1.0

        return p

    @staticmethod
    def forward(x, target, p):
        if target == 'w':
            if p == 1:  # if qb == 1
                w_hat = DoReFaSTEForW.apply(x, p)
            else:
                w_tanh = torch.tanh(x)
                w_sc = (w_tanh / torch.max(torch.abs(w_tanh)).detach() + 1.0) / 2.0
                w_hat = 2.0 * DoReFaSTEForW.apply(w_sc, p) - 1.0

            return w_hat
        else:
            assert target == 'a'

            x_sc = torch.clamp(x, min=0, max=1.0)
            x_hat = DoReFaSTEForA.apply(x_sc, p)

            return x_hat
    # Backpropagation will be conducted by chain rule

    @staticmethod
    def quant(x, target, p):
        if target == 'w':
            if p == 1:  # if qb == 1
                w_bar = torch.sign(x)
            else:
                w_tanh = torch.tanh(x)
                w_sc = (w_tanh / torch.max(torch.abs(w_tanh)).detach() + 1.0) / 2.0
                w_bar = torch.round(p * w_sc)

            return w_bar
        else:
            assert target == 'a'

            x_sc = torch.clamp(x, min=0, max=1.0)
            x_bar = torch.round(p * x_sc)

            return x_bar
##

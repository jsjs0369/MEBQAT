import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

"""
Written referring to:
    https://github.com/deJQK/AdaBits
    https://github.com/SHI-Labs/Any-Precision-DNNs
    https://github.com/KwangHoonAn/PACT
"""

class PACTSTE(autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, alpha):
        ctx.save_for_backward(x, alpha)
        y = torch.clamp(x, min=0, max=alpha.item())
        y = torch.round(y * ((2 ** bits - 1) / alpha)) / ((2 ** bits - 1) / alpha)

        return y

    @staticmethod
    def backward(ctx, dy):
        x, alpha = ctx.saved_tensors
        indicate_lower_bound = x < 0
        indicate_upper_bound = x > alpha
        x_range = ~(indicate_lower_bound | indicate_upper_bound)

        dx = dy * x_range.float()
        dalpha = torch.sum(dy * torch.ge(x, alpha).float()).view(-1)

        return dx, None, dalpha

class SignSTE(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.sign(x)

        return y

    @staticmethod
    def backward(ctx, dy):
        return dy

class SATSTE(autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, modified=False):
        assert torch.all(x >= 0) and torch.all(x <= 1)
        if not modified:    # Original DoReFa-Net
            ei = (1 << bits) - 1
            y = torch.round(ei * x).div_(ei)
        else:               # Modified DoReFa-Net
            ei_hat = 1 << bits
            y = torch.floor(ei_hat * x).clamp_(max=ei_hat - 1).div_(ei_hat)
        assert torch.all(y >= 0) and torch.all(y <= 1)

        return y

    @staticmethod
    def backward(ctx, dy):
        return dy, None, None

class LSQSTE(autograd.Function):
    @staticmethod
    def forward(ctx, x, scale_factor, scale_factor_grad, n, p):
        ctx.save_for_backward(x, scale_factor)
        ctx.others = scale_factor_grad, n, p

        y = x / (scale_factor + 1e-6)
        y = F.hardtanh(y, min_val=n, max_val=p)
        y = torch.round(y)
        y = y * scale_factor    

        return y

    @staticmethod
    def backward(ctx, dy):
        x, scale_factor = ctx.saved_tensors
        scale_factor_grad, n, p = ctx.others

        x_sc = x / (scale_factor + 1e-6)

        indicate_small = (x_sc < n).float()
        indicate_big = (x_sc > p).float()
        indicate_mid = 1.0 - indicate_small - indicate_big

        dx = dy * indicate_mid
        dscale_factor = (
            (indicate_small * n + indicate_big * p + indicate_mid * (torch.round(x_sc) - x_sc)) * dy * scale_factor_grad).sum().unsqueeze(dim=0)

        return dx, dscale_factor, None, None, None

class WeightQuantizer(nn.Module):
    def __init__(
        self,
        method,
        bits,
        **kwargs):
        super().__init__()

        # Attribute
        self.method = method
        self.bits = bits        

        if self.bits is not None:
            # [CVPR '20] AdaBits: Neural Network Quantization with Adaptive Bit-widths
            if self.method in ['SAT-originalW', 'SAT-modifiedW']:
                assert self.bits != 1                
                # Extra attribute
                assert 'scale' in kwargs
                self.scale = kwargs['scale']
                if self.scale:
                    assert 'out_features' in kwargs
                    self.out_features = kwargs['out_features']
            # [AAAI '21] Any-Precision Deep Neural Networks
            elif self.method == 'Yu21': 
                pass
            # [ICLR '20] Learned Step size Quantization
            elif self.method == 'LSQ':
                assert self.bits != 1
                # Extra attribute
                self.n = (-1.0) * (2.0 ** (bits - 1.0))
                self.p = (2.0 ** (bits - 1.0)) - 1.0
                assert 'weight' in kwargs
                self.scale_factor_grad = 1.0 / math.sqrt(kwargs['weight'].numel() * self.p)
                # Parameter
                self.scale_factor = nn.Parameter(
                    2.0 * torch.mean(torch.abs(kwargs['weight'])) / math.sqrt(self.p))
            # [arXiv '21] One Model for All Quantization: A Quantized Network Supporting Hot-Swap Bit-Width Adjustment
            elif self.method == 'Sun21':
                pass

    def forward(self, x):   # x: weight, y: quantized weight
        if self.bits == None:
            y = nn.Identity()(x)
        else:
            if self.method == 'SAT-originalW':
                y = torch.tanh(x) / torch.max(torch.abs(torch.tanh(x)))
                y.add_(1.0).div_(2.0)
                y = SATSTE.apply(y, self.bits)
                y.mul_(2.0).sub_(1.0)
                if self.scale:
                    scale_factor = 1.0 / (self.out_features) ** 0.5
                    scale_factor /= torch.std(y.detach())
                    y.mul_(scale_factor)
            elif self.method == 'SAT-modifiedW':
                y = torch.tanh(x) / torch.max(torch.abs(torch.tanh(x)))
                y.add_(1.0).div_(2.0)
                y = SATSTE.apply(y, self.bits, True)
                y.mul_(2.0).sub_(1.0)
                if self.scale:
                    scale_factor = 1.0 / (self.out_features) ** 0.5
                    scale_factor /= torch.std(y.detach())
                    y.mul_(scale_factor)
            elif self.method == 'Yu21':
                if self.bits == 1:
                    scale_factor = torch.mean(torch.abs(x)).detach()    # E
                    y = SignSTE.apply(x / scale_factor)
                    y.mul_(scale_factor)
                else:
                    y = torch.tanh(x) / torch.max(torch.abs(torch.tanh(x)))
                    y.add_(1.0).div_(2.0)
                    y = SATSTE.apply(y, self.bits)
                    y.mul_(2.0).sub_(1.0)
                    scale_factor = torch.mean(torch.abs(x)).detach()    # E
                    y.mul_(scale_factor)
            elif self.method == 'LSQ':
                y = LSQSTE.apply(x, self.scale_factor, self.scale_factor_grad, self.n, self.p)
            elif self.method == 'Sun21':    # Same as DoReFa-Net
                if self.bits == 1:
                    scale_factor = torch.mean(torch.abs(x)).detach()    # E
                    y = SignSTE.apply(x / scale_factor)
                    y.mul_(scale_factor)
                else:
                    y = torch.tanh(x) / torch.max(torch.abs(torch.tanh(x)))
                    y.add_(1.0).div_(2.0)
                    y = SATSTE.apply(y, self.bits)
                    y.mul_(2.0).sub_(1.0)

        return y

class ActivationQuantizer(nn.Module):
    def __init__(
        self,
        method,
        bits,
        **kwargs):
        super().__init__()

        # Attribute
        self.method = method
        self.bits = bits        

        if self.bits is not None:
            if self.method in ['SAT-originalW', 'SAT-modifiedW']:
                assert self.bits != 1
                # Parameter
                self.alpha = nn.Parameter(torch.tensor(8.0))
            elif self.method == 'Yu21':
                pass
            elif self.method == 'LSQ':
                assert self.bits != 1
                # Extra attribute
                self.n = 0
                self.p = 2.0 ** bits
                self.scale_factor_grad = None
                # Parameter
                self.scale_factor = nn.Parameter(torch.Tensor())
            elif self.method == 'Sun21':
                # Parameter
                self.alpha = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):   # x: activation, y: quantized activation
        if self.bits == None:
            y = nn.Identity()(x)
        else:
            if self.method in ['SAT-originalW', 'SAT-modifiedW']:
                alpha = torch.abs(self.alpha)
                y = torch.relu(x)
                y = torch.where(y < alpha, y, alpha)
                y.div_(alpha)
                y = SATSTE.apply(y, self.bits)
                y.mul_(alpha)
            elif self.method == 'Yu21':
                y = torch.clamp(x, min=0, max=1)
                y = SATSTE.apply(y, self.bits)
            elif self.method == 'LSQ':
                if self.scale_factor_grad is None:
                    self.scale_factor_grad = 1.0 / math.sqrt(x.numel() * self.p)
                if self.scale_factor.nelement() == 0:
                    self.scale_factor = nn.Parameter(
                        2.0 * torch.mean(torch.abs(x)) / math.sqrt(self.p))
                y = LSQSTE.apply(x, self.scale_factor, self.scale_factor_grad, self.n, self.p)
            elif self.method == 'Sun21':        # Same as PACT
                y = PACTSTE.apply(x, self.bits, self.alpha)

        return y

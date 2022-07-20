import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_, normal_, zeros_
# From own code
from quant import WeightQuantizer, ActivationQuantizer
from util import make_divisible

"""
Written referring to:
    https://pytorch.org/hub/pytorch_vision_resnet/
    https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
    https://github.com/SHI-Labs/Any-Precision-DNNs
    https://github.com/kuangliu/pytorch-cifar
"""

## Basic modules
class Conv2d(nn.Module):
    def __init__(
        self,
        q_method,
        q_bits_w,
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=True): 
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv.reset_parameters()

        q_kwargs = {}
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            q_kwargs['scale'] = False
        elif q_method == 'LSQ':
            q_kwargs['weight'] = self.conv.weight.data
        self.quantizer_w = WeightQuantizer(q_method, q_bits_w, **q_kwargs)

    def forward(self, x):
        q_w = self.quantizer_w(self.conv.weight)
        y = F.conv2d(
            x, q_w,
            bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)

        return y

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        track_running_stats):
        super().__init__(
            num_features,
            track_running_stats=track_running_stats)

        nn.init.ones_(self.weight)  # gamma
        nn.init.zeros_(self.bias)   # beta

class BatchNorm2dOnlyBeta(nn.BatchNorm2d):  # BatchNorm2d optimizing bias (beta) only
    def __init__(
        self,
        num_features,
        track_running_stats):
        super().__init__(
            num_features,
            track_running_stats=track_running_stats)

        nn.init.zeros_(self.bias)   # beta

        # gamma, fixed to 1
        delattr(self, 'weight')
        self.register_buffer(
            'weight',
            torch.ones(num_features))

class ActivationFunction(nn.Module):
    def __init__(
        self,
        q_method,
        q_bits_a,
        activation_layer=nn.ReLU,
        **kwargs):
        super().__init__()

        self.act_func = activation_layer(**kwargs)

        q_kwargs = {}
        self.quantizer_a = ActivationQuantizer(q_method, q_bits_a, **q_kwargs)

    def forward(self, x):
        a = self.act_func(x)
        q_a = self.quantizer_a(a)
        y = q_a

        return y

class Linear(nn.Module):
    def __init__(
        self,
        q_method,
        q_bits_w,
        in_features, 
        out_features, 
        bias=True):
        super().__init__()

        self.fc = nn.Linear(
            in_features, out_features,
            bias=bias)
        self.fc.reset_parameters()

        q_kwargs = {}
        if q_method in ['SAT-originalW', 'SAT-modifiedW']:
            q_kwargs['scale'] = True
            q_kwargs['out_features'] = self.fc.out_features
        elif q_method == 'LSQ':
            q_kwargs['weight'] = self.fc.weight.data
        self.quantizer_w = WeightQuantizer(q_method, q_bits_w, **q_kwargs)

    def forward(self, x):
        q_w = self.quantizer_w(self.fc.weight)
        y = F.linear(
            x, q_w,
            bias=self.fc.bias)

        return y
##

## ResNet18/50 (vs. One model for all quantization/AdaBits, on ImageNet, can be pretrained)
def conv3x3(
    q_method,
    q_bits_w,
    in_planes, 
    out_planes, 
    stride=1, 
    groups=1, 
    dilation=1):
    return Conv2d(
        q_method, q_bits_w,
        in_planes, out_planes,
        kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(
    q_method,
    q_bits_w,
    in_planes, 
    out_planes, 
    stride=1):
    return Conv2d(
        q_method, q_bits_w,
        in_planes, out_planes,
        kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        q_method,
        q_bits_w_list, q_bits_a_list,
        track_running_stats,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Work for quantization
        q_bits_w_iter, q_bits_a_iter = iter(q_bits_w_list), iter(q_bits_a_list) 

        q_bits_w = next(q_bits_w_iter)
        self.conv1 = conv3x3(
            q_method, q_bits_w,
            inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes, track_running_stats)
        q_bits_a = next(q_bits_a_iter)
        self.relu1 = ActivationFunction(q_method, q_bits_a, inplace=True)

        q_bits_w = next(q_bits_w_iter)
        self.conv2 = conv3x3(
            q_method, q_bits_w,            
            planes, planes)
        self.bn2 = norm_layer(planes, track_running_stats)
        self.downsample = downsample
        q_bits_a = next(q_bits_a_iter)        
        self.relu2 = ActivationFunction(q_method, q_bits_a, inplace=True)

        self.stride = stride

    def forward(self, x):
        identity = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        if self.downsample is not None:
            identity = self.downsample(x)
        y += identity
        y = self.relu2(y)

        return y

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        q_method,
        q_bits_w_list, q_bits_a_list,
        track_running_stats,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        # Work for quantization
        q_bits_w_iter, q_bits_a_iter = iter(q_bits_w_list), iter(q_bits_a_list)     

        q_bits_w = next(q_bits_w_iter)
        self.conv1 = conv1x1(
            q_method, q_bits_w,
            inplanes, width)
        self.bn1 = norm_layer(width, track_running_stats)
        q_bits_a = next(q_bits_a_iter)
        self.relu1 = ActivationFunction(q_method, q_bits_a, inplace=True)

        q_bits_w = next(q_bits_w_iter)
        self.conv2 = conv3x3(
            q_method, q_bits_w,
            width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, track_running_stats)
        q_bits_a = next(q_bits_a_iter)
        self.relu2 = ActivationFunction(q_method, q_bits_a, inplace=True)

        q_bits_w = next(q_bits_w_iter)
        self.conv3 = conv1x1(
            q_method, q_bits_w,
            width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, track_running_stats)
        self.downsample = downsample
        q_bits_a = next(q_bits_a_iter)
        self.relu3 = ActivationFunction(q_method, q_bits_a, inplace=True)

        self.stride = stride

    def forward(self, x):
        identity = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)

        y = self.conv3(y)
        y = self.bn3(y)
        if self.downsample is not None:
            identity = self.downsample(x)
        y += identity
        y = self.relu3(y)

        return y

class ResNet18(nn.Module):
    def __init__(
        self,
        q_method,
        q_bits_w_list, q_bits_a_list,
        track_running_stats,
        num_classes=1000,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        pretrained=True):
        super().__init__()

        if norm_layer is None:
            norm_layer = BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}")
        self.groups = groups
        self.base_width = width_per_group

        # ResNet18
        block = BasicBlock
        layers = [2, 2, 2, 2]

        # Work for BN
        self.track_running_stats = track_running_stats

        # Work for quantization
        self.q_method = q_method
        self.q_bits_w_iter, self.q_bits_a_iter = iter(q_bits_w_list), iter(q_bits_a_list)

        q_bits_w = next(self.q_bits_w_iter)
        self.conv1 = Conv2d(
            q_method, q_bits_w,
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes, self.track_running_stats)
        q_bits_a = next(self.q_bits_a_iter)
        self.relu1 = ActivationFunction(q_method, q_bits_a, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        q_bits_w = next(self.q_bits_w_iter)        
        self.fc = Linear(
            q_method, q_bits_w,
            512 * block.expansion, num_classes)

        if pretrained:
            self.load_state_dict(
                torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth'), 
                strict=False)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            q_bits_w = next(self.q_bits_w_iter)
            downsample = nn.Sequential(
                conv1x1(
                    self.q_method, q_bits_w,
                    self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, self.track_running_stats))

        layers = []
        q_bits_w1, q_bits_a1 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)
        q_bits_w2, q_bits_a2 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)
        layers.append(
            block(
                self.q_method, [q_bits_w1, q_bits_w2], [q_bits_a1, q_bits_a2], 
                self.track_running_stats,
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            q_bits_w1, q_bits_a1 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)
            q_bits_w2, q_bits_a2 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)
            layers.append(
                block(
                    self.q_method, [q_bits_w1, q_bits_w2], [q_bits_a1, q_bits_a2], 
                    self.track_running_stats,
                    self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avgpool(y)

        y = torch.flatten(y, 1)
        y = self.fc(y)

        return y

    def forward(self, x):
        return self._forward_impl(x)

class ResNet50(nn.Module):
    def __init__(
        self,
        q_method,
        q_bits_w_list, q_bits_a_list,
        track_running_stats,
        num_classes=1000,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        pretrained=True):
        super().__init__()

        if norm_layer is None:
            norm_layer = BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}")
        self.groups = groups
        self.base_width = width_per_group

        # ResNet50
        block = Bottleneck
        layers = [3, 4, 6, 3]

        # Work for BN
        self.track_running_stats = track_running_stats

        # Work for quantization
        self.q_method = q_method
        self.q_bits_w_iter, self.q_bits_a_iter = iter(q_bits_w_list), iter(q_bits_a_list)

        q_bits_w = next(self.q_bits_w_iter)
        self.conv1 = Conv2d(
            q_method, q_bits_w,
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes, self.track_running_stats)
        q_bits_a = next(self.q_bits_a_iter)
        self.relu1 = ActivationFunction(q_method, q_bits_a, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        q_bits_w = next(self.q_bits_w_iter)        
        self.fc = Linear(
            q_method, q_bits_w,
            512 * block.expansion, num_classes)

        if pretrained:
            self.load_state_dict(
                torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet50-0676ba61.pth'), 
                strict=False)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            q_bits_w = next(self.q_bits_w_iter)
            downsample = nn.Sequential(
                conv1x1(
                    self.q_method, q_bits_w,
                    self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, self.track_running_stats))

        layers = []
        q_bits_w1, q_bits_a1 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)
        q_bits_w2, q_bits_a2 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)
        q_bits_w3, q_bits_a3 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)
        layers.append(
            block(
                self.q_method, [q_bits_w1, q_bits_w2, q_bits_w3], [q_bits_a1, q_bits_a2, q_bits_a3], 
                self.track_running_stats,
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            q_bits_w1, q_bits_a1 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)
            q_bits_w2, q_bits_a2 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)
            q_bits_w3, q_bits_a3 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)
            layers.append(
                block(
                    self.q_method, [q_bits_w1, q_bits_w2, q_bits_w3], [q_bits_a1, q_bits_a2, q_bits_a3], 
                    self.track_running_stats,
                    self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avgpool(y)

        y = torch.flatten(y, 1)
        y = self.fc(y)

        return y

    def forward(self, x):
        return self._forward_impl(x)
##

## MobileNetV2 for ImageNet/CIFAR (vs. AdaBits/Any-precision DNN, on ImageNet/CIFAR10, can/cannot be pretrained)
class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        q_method,
        q_bits_w, q_bits_a,
        track_running_stats,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        norm_layer=BatchNorm2d,
        activation_layer=nn.ReLU,
        dilation=1,
        inplace=True,
        bias=None):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            Conv2d(
                q_method, q_bits_w,
                in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels, track_running_stats))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(ActivationFunction(
                q_method, q_bits_a,
                activation_layer=activation_layer, **params))
        super().__init__(*layers)

        self.out_channels = out_channels

class InvertedResidual(nn.Module):
    def __init__(
        self,
        q_method,
        q_bits_w_list, q_bits_a_list,
        track_running_stats,        
        inp, 
        oup, 
        stride, 
        expand_ratio, 
        norm_layer=None):
        super().__init__()

        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        # Work for quantization
        q_bits_w_iter, q_bits_a_iter = iter(q_bits_w_list), iter(q_bits_a_list)     

        layers = []
        if expand_ratio != 1:
            q_bits_w = next(q_bits_w_iter)
            q_bits_a = next(q_bits_a_iter)
            # pw
            layers.append(ConvNormActivation(
                q_method, q_bits_w, q_bits_a,
                track_running_stats,
                inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        q_bits_w1 = next(q_bits_w_iter)
        q_bits_w2 = next(q_bits_w_iter)
        q_bits_a1 = next(q_bits_a_iter)
        layers.extend([
                # dw
                ConvNormActivation(
                    q_method, q_bits_w1, q_bits_a1,
                    track_running_stats,
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer),
                # pw-linear
                Conv2d(
                    q_method, q_bits_w2,
                    hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup, track_running_stats)])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2ForImageNet(nn.Module):
    def __init__(
        self,
        q_method,
        q_bits_w_list, q_bits_a_list,
        track_running_stats,
        num_classes=1000,
        width_mult=1.0,
        inverted_residual_setting=None,
        round_nearest=8,
        block=None,
        norm_layer=None,
        dropout=0.2,
        pretrained=True):
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # Work for quantization
        q_bits_w_iter, q_bits_a_iter = iter(q_bits_w_list), iter(q_bits_a_list)

        # building first layer
        input_channel = make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        q_bits_w = next(q_bits_w_iter)
        q_bits_a = next(q_bits_a_iter)
        features = [ConvNormActivation(
            q_method, q_bits_w, q_bits_a,
            track_running_stats,
            3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                if t != 1:
                    q_bits_w1 = next(q_bits_w_iter)
                    q_bits_w2 = next(q_bits_w_iter)
                    q_bits_w3 = next(q_bits_w_iter)
                    q_bits_a1 = next(q_bits_a_iter)
                    q_bits_a2 = next(q_bits_a_iter)
                    features.append(block(
                        q_method, [q_bits_w1, q_bits_w2, q_bits_w3], [q_bits_a1, q_bits_a2], 
                        track_running_stats,
                        input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                else:
                    q_bits_w1 = next(q_bits_w_iter)
                    q_bits_w2 = next(q_bits_w_iter)
                    q_bits_a = next(q_bits_a_iter)
                    features.append(block(
                        q_method, [q_bits_w1, q_bits_w2], [q_bits_a], 
                        track_running_stats,
                        input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        q_bits_w = next(q_bits_w_iter)
        q_bits_a = next(q_bits_a_iter)        
        features.append(
            ConvNormActivation(
                q_method, q_bits_w, q_bits_a,
                track_running_stats,
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        q_bits_w = next(q_bits_w_iter)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            Linear(
                q_method, q_bits_w,
                self.last_channel, num_classes))

        if pretrained:
            self.load_state_dict(
                torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'), 
                strict=False)

    def _forward_impl(self, x):
        y = self.features(x)
        y = F.adaptive_avg_pool2d(y, (1, 1))
        y = torch.flatten(y, 1)
        y = self.classifier(y)

        return y

    def forward(self, x):
        return self._forward_impl(x)

class MobileNetV2ForCIFAR(nn.Module):
    def __init__(
        self,
        q_method,
        q_bits_w_list, q_bits_a_list,
        track_running_stats,
        num_classes=1000,
        width_mult=1.0,
        inverted_residual_setting=None,
        round_nearest=8,
        block=None,
        norm_layer=None,
        dropout=0.2,
        pretrained=False):
        super().__init__()

        assert not pretrained

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 1],  # Change stride 2 -> 1 for CIFAR
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # Work for quantization
        q_bits_w_iter, q_bits_a_iter = iter(q_bits_w_list), iter(q_bits_a_list)

        # building first layer
        input_channel = make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        q_bits_w = next(q_bits_w_iter)
        q_bits_a = next(q_bits_a_iter)
        # Change stride 2 -> 1 for CIFAR
        features = [ConvNormActivation(
            q_method, q_bits_w, q_bits_a,
            track_running_stats,
            3, input_channel, stride=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                if t != 1:
                    q_bits_w1 = next(q_bits_w_iter)
                    q_bits_w2 = next(q_bits_w_iter)
                    q_bits_w3 = next(q_bits_w_iter)
                    q_bits_a1 = next(q_bits_a_iter)
                    q_bits_a2 = next(q_bits_a_iter)
                    features.append(block(
                        q_method, [q_bits_w1, q_bits_w2, q_bits_w3], [q_bits_a1, q_bits_a2], 
                        track_running_stats,
                        input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                else:
                    q_bits_w1 = next(q_bits_w_iter)
                    q_bits_w2 = next(q_bits_w_iter)
                    q_bits_a = next(q_bits_a_iter)
                    features.append(block(
                        q_method, [q_bits_w1, q_bits_w2], [q_bits_a], 
                        track_running_stats,
                        input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        q_bits_w = next(q_bits_w_iter)
        q_bits_a = next(q_bits_a_iter)        
        features.append(
            ConvNormActivation(
                q_method, q_bits_w, q_bits_a,
                track_running_stats,
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        q_bits_w = next(q_bits_w_iter)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            Linear(
                q_method, q_bits_w,
                self.last_channel, num_classes))

    def _forward_impl(self, x):
        y = self.features(x)
        y = F.adaptive_avg_pool2d(y, (1, 1))    # Change pooling kernel_size 7 -> 4 for CIFAR
        y = torch.flatten(y, 1)
        y = self.classifier(y)

        return y

    def forward(self, x):
        return self._forward_impl(x)
##

## Pre-activation ResNet50 (vs. Any-precision DNN, on ImageNet, cannot be pretrained)
class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, 
        q_method,
        q_bits_w_list, q_bits_a_list,
        track_running_stats,
        in_planes, 
        out_planes, 
        stride=1, 
        downsample=None,
        norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = BatchNorm2d

        # Work for quantization
        q_bits_w_iter, q_bits_a_iter = iter(q_bits_w_list), iter(q_bits_a_list) 

        self.bn1 = norm_layer(in_planes, track_running_stats)
        q_bits_a = next(q_bits_a_iter)
        self.relu1 = ActivationFunction(q_method, q_bits_a, inplace=True)
        q_bits_w = next(q_bits_w_iter)
        self.conv1 = Conv2d(
            q_method, q_bits_w,
            in_planes, out_planes, kernel_size=1, stride=1, bias=False)

        self.bn2 = norm_layer(out_planes, track_running_stats)
        q_bits_a = next(q_bits_a_iter)
        self.relu2 = ActivationFunction(q_method, q_bits_a, inplace=True)
        q_bits_w = next(q_bits_w_iter)
        self.conv2 = Conv2d(
            q_method, q_bits_w,
            out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = norm_layer(out_planes, track_running_stats)
        q_bits_a = next(q_bits_a_iter)
        self.relu3 = ActivationFunction(q_method, q_bits_a, inplace=True)
        q_bits_w = next(q_bits_w_iter)
        self.conv3 = Conv2d(
            q_method, q_bits_w,
            out_planes, out_planes * self.expansion, kernel_size=1, stride=1, bias=False)

        self.downsample = downsample

    def forward(self, x):
        y = self.bn1(x)
        y = self.relu1(y)
        y = self.conv1(y)

        y = self.bn2(y)
        y = self.relu2(y)
        y = self.conv2(y)

        y = self.bn3(y)
        y = self.relu3(y)
        y = self.conv3(y)

        shortcut = self.downsample(x) if self.downsample is not None else x
        y += shortcut

        return y

class PreActResNet50(nn.Module):
    def __init__(
        self,
        q_method,
        q_bits_w_list, q_bits_a_list,
        track_running_stats,        
        norm_layer=None,
        num_classes=1000,
        pretrained=False):
        super().__init__()

        assert not pretrained

        if norm_layer is None:
            norm_layer = BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64

        # Pre-activation ResNet50
        block = PreActBottleneck
        layers = [3, 4, 6, 3]

        # Work for BN
        self.track_running_stats = track_running_stats

        # Work for quantization
        self.q_method = q_method
        self.q_bits_w_iter, self.q_bits_a_iter = iter(q_bits_w_list), iter(q_bits_a_list)

        q_bits_w = next(self.q_bits_w_iter)
        self.conv1 = Conv2d(
            q_method, q_bits_w,
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn_last = self._norm_layer(512 * block.expansion, track_running_stats)
        q_bits_a = next(self.q_bits_a_iter)
        self.relu_last = ActivationFunction(q_method, q_bits_a, inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        q_bits_w = next(self.q_bits_w_iter)
        self.fc = Linear(
            q_method, q_bits_w,
            512 * block.expansion, num_classes)

    def _make_layer(
        self, 
        block, 
        planes, 
        blocks, 
        stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            q_bits_w = next(self.q_bits_w_iter)
            downsample = nn.Sequential(
                Conv2d(
                    self.q_method, q_bits_w,
                    self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self._norm_layer(planes * block.expansion, self.track_running_stats))

        layers = []
        q_bits_w1, q_bits_a1 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)
        q_bits_w2, q_bits_a2 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)
        q_bits_w3, q_bits_a3 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)        
        layers.append(block(
            self.q_method, [q_bits_w1, q_bits_w2, q_bits_w3], [q_bits_a1, q_bits_a2, q_bits_a3], 
            self.track_running_stats,
            self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            q_bits_w1, q_bits_a1 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)
            q_bits_w2, q_bits_a2 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)
            q_bits_w3, q_bits_a3 = next(self.q_bits_w_iter), next(self.q_bits_a_iter)             
            layers.append(block(
                self.q_method, [q_bits_w1, q_bits_w2, q_bits_w3], [q_bits_a1, q_bits_a2, q_bits_a3], 
                self.track_running_stats,                
                self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        y = self.bn_last(y)
        y = self.relu_last(y)
        y = self.avgpool(y)

        y = torch.flatten(y, 1)
        y = self.fc(y)

        return y
##

## Pre-activation ResNet20 (vs. Any-precision DNN, on CIFAR10, cannot be pretrained)
class PreActBasicBlock(nn.Module):
    def __init__(
        self, 
        q_method,
        q_bits_w_list, q_bits_a_list,
        track_running_stats,
        in_planes, 
        out_planes, 
        stride=1,
        norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = BatchNorm2d

        # Work for quantization
        q_bits_w_iter, q_bits_a_iter = iter(q_bits_w_list), iter(q_bits_a_list) 

        self.bn1 = norm_layer(in_planes, track_running_stats)
        q_bits_a = next(q_bits_a_iter)
        self.relu1 = ActivationFunction(q_method, q_bits_a, inplace=True)
        q_bits_w = next(q_bits_w_iter)
        self.conv1 = Conv2d(
            q_method, q_bits_w,
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = norm_layer(out_planes, track_running_stats)
        q_bits_a = next(q_bits_a_iter)
        self.relu2 = ActivationFunction(q_method, q_bits_a, inplace=True)
        q_bits_w = next(q_bits_w_iter)
        self.conv2 = Conv2d(
            q_method, q_bits_w,
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.skip_conv = None
        if stride != 1:
            q_bits_w = next(q_bits_w_iter)
            self.skip_conv = Conv2d(
                q_method, q_bits_w,
                in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.skip_bn = norm_layer(out_planes, track_running_stats)

    def forward(self, x):
        y = self.bn1(x)
        y = self.relu1(y)

        if self.skip_conv is not None:
            shortcut = self.skip_conv(y)
            shortcut = self.skip_bn(shortcut)
        else:
            shortcut = x

        y = self.conv1(y)

        y = self.bn2(y)
        y = self.relu2(y)
        y = self.conv2(y)

        y += shortcut

        return y

class PreActResNet20(nn.Module):
    def __init__(
        self, 
        q_method,
        q_bits_w_list, q_bits_a_list,
        track_running_stats,
        norm_layer=None, 
        num_classes=10,
        expand=5,
        pretrained=False):
        super().__init__()

        assert not pretrained

        if norm_layer is None:
            norm_layer = BatchNorm2d
        self.expand = expand
        ep = self.expand

        # Pre-activation ResNet20
        block = PreActBasicBlock
        num_units = [3, 3, 3]

        # Work for quantization
        q_bits_w_iter, q_bits_a_iter = iter(q_bits_w_list), iter(q_bits_a_list) 

        q_bits_w = next(q_bits_w_iter)
        self.conv1 = Conv2d(
            q_method, q_bits_w,
            3, 16 * ep, kernel_size=3, stride=1, padding=1, bias=False)

        strides = [1] * num_units[0] + [2] + [1] * (num_units[1] - 1) + [2] + [1] * (num_units[2] - 1)
        channels = [16 * ep] * num_units[0] + [32 * ep] * num_units[1] + [64 * ep] * num_units[2]
        in_planes = 16 * ep
        self.layers = nn.ModuleList()
        for stride, channel in zip(strides, channels):
            if stride != 1:
                q_bits_w1, q_bits_a1 = next(q_bits_w_iter), next(q_bits_a_iter)
                q_bits_w2, q_bits_a2 = next(q_bits_w_iter), next(q_bits_a_iter)
                q_bits_w3 = next(q_bits_w_iter)
                self.layers.append(block(
                    q_method, [q_bits_w1, q_bits_w2, q_bits_w3], [q_bits_a1, q_bits_a2],
                    track_running_stats,
                    in_planes, channel, stride))            
            else:
                q_bits_w1, q_bits_a1 = next(q_bits_w_iter), next(q_bits_a_iter)
                q_bits_w2, q_bits_a2 = next(q_bits_w_iter), next(q_bits_a_iter)
                self.layers.append(block(
                    q_method, [q_bits_w1, q_bits_w2], [q_bits_a1, q_bits_a2],
                    track_running_stats,
                    in_planes, channel, stride))
            in_planes = channel

        self.bn_last = norm_layer(64 * ep, track_running_stats)

        q_bits_w = next(q_bits_w_iter)
        self.fc = Linear(
            q_method, q_bits_w,
            64 * ep, num_classes)

    def forward(self, x):
        y = self.conv1(x)

        for layer in self.layers:
            y = layer(y)

        y = self.bn_last(y)

        y = y.mean(dim=2).mean(dim=2)
        y = self.fc(y)

        return y
##

# 8-layer CNN (vs. Any-precision DNN, on SVHN, cannot be pretrained)
class CNNForSVHN(nn.Module):
    def __init__(
        self, 
        q_method,
        q_bits_w_list, q_bits_a_list,
        track_running_stats,
        norm_layer=None, 
        num_classes=10, 
        expand=8,
        pretrained=False):
        super().__init__()

        assert not pretrained

        if norm_layer is None:
            norm_layer = BatchNorm2d
        self.expand = expand
        ep = self.expand

        # Work for quantization
        q_bits_w_iter, q_bits_a_iter = iter(q_bits_w_list), iter(q_bits_a_list) 

        layers = []

        q_bits_w, q_bits_a = next(q_bits_w_iter), next(q_bits_a_iter)
        layers.extend([
            Conv2d(
                q_method, q_bits_w,
                3, ep * 6, 5, padding=0, bias=True),
            nn.MaxPool2d(2),
            ActivationFunction(q_method, q_bits_a, inplace=True)])
        # 18
        q_bits_w, q_bits_a = next(q_bits_w_iter), next(q_bits_a_iter)
        layers.extend([
            Conv2d(
                q_method, q_bits_w,
                ep * 6, ep * 8, 3, padding=1, bias=False),
            norm_layer(ep * 8, track_running_stats),
            ActivationFunction(q_method, q_bits_a, inplace=True)])

        q_bits_w, q_bits_a = next(q_bits_w_iter), next(q_bits_a_iter)
        layers.extend([
            Conv2d(
                q_method, q_bits_w,
                ep * 8, ep * 8, 3, padding=1, bias=False),
            norm_layer(ep * 8, track_running_stats),
            nn.MaxPool2d(2),
            ActivationFunction(q_method, q_bits_a, inplace=True)])
        # 9
        q_bits_w, q_bits_a = next(q_bits_w_iter), next(q_bits_a_iter)
        layers.extend([
            Conv2d(
                q_method, q_bits_w,
                ep * 8, ep * 16, 3, padding=0, bias=False),
            norm_layer(ep * 16, track_running_stats),
            ActivationFunction(q_method, q_bits_a, inplace=True)])
        # 7
        q_bits_w, q_bits_a = next(q_bits_w_iter), next(q_bits_a_iter)
        layers.extend([
            Conv2d(
                q_method, q_bits_w,
                ep * 16, ep * 16, 3, padding=1, bias=False),
            norm_layer(ep * 16, track_running_stats),
            ActivationFunction(q_method, q_bits_a, inplace=True)])

        q_bits_w, q_bits_a = next(q_bits_w_iter), next(q_bits_a_iter)
        layers.extend([
            Conv2d(
                q_method, q_bits_w,
                ep * 16, ep * 16, 3, padding=0, bias=False),
            norm_layer(ep * 16, track_running_stats),
            ActivationFunction(q_method, q_bits_a, inplace=True)])
        # 5
        q_bits_w, q_bits_a = next(q_bits_w_iter), next(q_bits_a_iter)
        layers.extend([
            nn.Dropout(0.5),
            Conv2d(
                q_method, q_bits_w,
                ep * 16, ep * 64, 5, padding=0, bias=False),
            ActivationFunction(q_method, q_bits_a, inplace=True)])

        self.layers = nn.Sequential(*layers)

        q_bits_w = next(q_bits_w_iter)
        self.fc = Linear(
            q_method, q_bits_w,
            ep * 64, 10)

    def forward(self, x):
        y = self.layers(x)

        y = y.view([y.shape[0], -1])
        y = self.fc(y)

        return y

# 5-layer CNN (vs. pure MAML and MAML + dedicated QAT, on Omniglot and MiniImageNet, cannot be pretrained)
class CNNForMAML(nn.Module):
    def __init__(
        self,
        q_method,
        q_bits_w_list, q_bits_a_list,
        track_running_stats,
        in_channels,
        hidden_channels,
        classifier_in_features,
        norm_layer=None,
        num_classes=20,
        pretrained=False):
        super().__init__()

        assert not pretrained

        if norm_layer is None:
            norm_layer = BatchNorm2dOnlyBeta

        # Work for quantization
        q_bits_w_iter, q_bits_a_iter = iter(q_bits_w_list), iter(q_bits_a_list)

        q_bits_w, q_bits_a = next(q_bits_w_iter), next(q_bits_a_iter)
        self.features1 = ConvNormActivation(
            q_method, q_bits_w, q_bits_a,
            track_running_stats,
            in_channels, hidden_channels, norm_layer=norm_layer, bias=True)

        q_bits_w, q_bits_a = next(q_bits_w_iter), next(q_bits_a_iter)
        self.features2 = ConvNormActivation(
            q_method, q_bits_w, q_bits_a,
            track_running_stats,
            hidden_channels, hidden_channels, norm_layer=norm_layer, bias=True)        

        q_bits_w, q_bits_a = next(q_bits_w_iter), next(q_bits_a_iter)
        self.features3 = ConvNormActivation(
            q_method, q_bits_w, q_bits_a,
            track_running_stats,
            hidden_channels, hidden_channels, norm_layer=norm_layer, bias=True)  

        q_bits_w, q_bits_a = next(q_bits_w_iter), next(q_bits_a_iter)
        self.features4 = ConvNormActivation(
            q_method, q_bits_w, q_bits_a,
            track_running_stats,
            hidden_channels, hidden_channels, norm_layer=norm_layer, bias=True)

        q_bits_w = next(q_bits_w_iter)
        self.classifier = Linear(
            q_method, q_bits_w,
            classifier_in_features, num_classes)

    def forward(self, x):
        # Default height (== width) of x: 28 / 84 for Omniglot / MiniImageNet
        y = self.features1(x)   
        y = F.max_pool2d(y, 2)  # Default height (== width) of y: 14 / 42 for Omniglot / MiniImageNet

        y = self.features2(y)
        y = F.max_pool2d(y, 2)  # Default height (== width) of y: 7 / 21 for Omniglot / MiniImageNet

        y = self.features3(y)
        y = F.max_pool2d(y, 2)  # Default height (== width) of y: 3 / 10 for Omniglot / MiniImageNet 

        y = self.features4(y)
        y = F.max_pool2d(y, 2)  # Default height (== width) of y: 1 / 5 for Omniglot / MiniImageNet

        y = y.view(y.size(0), -1)
        y = self.classifier(y)
  
        return y

# 4-layer CNN (vs. pure ProtoNet and ProtoNet + dedicated QAT, on Omniglot and MiniImageNet, cannot be pretrained)
class CNNForProtoNet(nn.Module):
    def __init__(
        self,
        q_method,
        q_bits_w_list, q_bits_a_list,
        track_running_stats,
        in_channels,
        hidden_channels=64,
        out_channels=64,
        norm_layer=None,
        pretrained=False):
        # Initialize class
        super().__init__()

        assert not pretrained

        if norm_layer is None:
            norm_layer = BatchNorm2d

        # Work for quantization
        q_bits_w_iter, q_bits_a_iter = iter(q_bits_w_list), iter(q_bits_a_list)

        q_bits_w, q_bits_a = next(q_bits_w_iter), next(q_bits_a_iter)
        self.features1 = ConvNormActivation(
            q_method, q_bits_w, q_bits_a,
            track_running_stats,
            in_channels, hidden_channels, norm_layer=norm_layer, bias=True)

        q_bits_w, q_bits_a = next(q_bits_w_iter), next(q_bits_a_iter)
        self.features2 = ConvNormActivation(
            q_method, q_bits_w, q_bits_a,
            track_running_stats,
            hidden_channels, hidden_channels, norm_layer=norm_layer, bias=True)        

        q_bits_w, q_bits_a = next(q_bits_w_iter), next(q_bits_a_iter)
        self.features3 = ConvNormActivation(
            q_method, q_bits_w, q_bits_a,
            track_running_stats,
            hidden_channels, hidden_channels, norm_layer=norm_layer, bias=True)  

        q_bits_w, q_bits_a = next(q_bits_w_iter), next(q_bits_a_iter)
        self.features4 = ConvNormActivation(
            q_method, q_bits_w, q_bits_a,
            track_running_stats,
            hidden_channels, out_channels, norm_layer=norm_layer, bias=True)

    def forward(self, x):
        # Default height (== width) of x: 28 / 84 for Omniglot / MiniImageNet
        y = self.features1(x)   
        y = F.max_pool2d(y, 2)  # Default height (== width) of y: 14 / 42 for Omniglot / MiniImageNet

        y = self.features2(y)
        y = F.max_pool2d(y, 2)  # Default height (== width) of y: 7 / 21 for Omniglot / MiniImageNet

        y = self.features3(y)
        y = F.max_pool2d(y, 2)  # Default height (== width) of y: 3 / 10 for Omniglot / MiniImageNet 

        y = self.features4(y)
        y = F.max_pool2d(y, 2)  # Default height (== width) of y: 1 / 5 for Omniglot / MiniImageNet

        y = y.view(y.size(0), -1)
  
        return y

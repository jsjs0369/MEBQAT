import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_, normal_, kaiming_uniform_, _calculate_fan_in_and_fan_out, uniform_, ones_, zeros_
# From own code(s)
from quant import LSQ, DoReFa
from util import make_divisible

## Common
class Conv2d(nn.Conv2d):
    def __init__(
        self,
        quant_scheme,
        qb_w, qb_a,
        in_channels,
        out_channels,
        kernel_size,
        init_w_method,
        bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        **kwargs):
        # Initialize class
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            dilation=dilation, groups=groups, 
            bias=bias)

        ## Initialize param(s)
        # Weight
        if init_w_method == 'trunc_normal':
            trunc_normal_(self.weight, std=0.02, a=-0.04, b=0.04)
            if bias:
                zeros_(self.bias)
        elif init_w_method == 'kaiming_uniform':
            kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                uniform_(self.bias, -bound, bound)
        ## Initialize param(s), end 

        ## Prepare QAT & quantization
        self.quant_scheme = quant_scheme
        self.qb_w, self.qb_a = qb_w, qb_a

        if quant_scheme is None:
            assert (qb_w == 32) and (qb_a == 32)
        else:
            assert qb_w < 32

        if quant_scheme == 'lsq':
            assert 'prev_act' in kwargs     # 'relu' or 'relu6' or 'swish' or 'etc'

            self.n_w, self.p_w, self.scale_grad_w, self.scale_w = LSQ.boot(qb_w, self.weight, 'w')            

            if qb_a < 32:
                self.n_a, self.p_a, self.scale_grad_a, self.scale_a = LSQ.boot(qb_a, None, 'a', prev_act=kwargs['prev_act'])
        elif quant_scheme == 'dorefa': 
            self.p_w = DoReFa.boot(qb_w)

            if qb_a < 32:
                self.p_a = DoReFa.boot(qb_a)
        else:
            assert quant_scheme is None
        ## Prepare QAT & quantization, end

    ## Aggregation
    def forward(self, x):
        if self.quant_scheme == 'lsq':
            return self.forward_lsq(x)
        elif self.quant_scheme == 'dorefa':
            return self.forward_dorefa(x)
        else:
            return super(Conv2d, self).forward(x)

    def quant_w(self):
        if self.quant_scheme == 'lsq':
            self.quant_w_lsq()
        elif self.quant_scheme == 'dorefa':
            self.quant_w_dorefa()
        else:
            pass
    ## Aggregation, end

    ## LSQ
    def forward_lsq(self, x):
        if self.training:
            w_used = LSQ.apply(self.weight, self.scale_w, self.scale_grad_w, self.n_w, self.p_w)

            if self.qb_a < 32:
                if self.scale_grad_a is None:
                    _, _, self.scale_grad_a, _ = LSQ.boot(self.qb_a, x, 'a', p=self.p_a)
                if self.scale_a.nelement() == 0:
                    _, _, _, self.scale_a = LSQ.boot(self.qb_a, x, 'a', p=self.p_a)
                a_used = LSQ.apply(x, self.scale_a, self.scale_grad_a, self.n_a, self.p_a)                
            else:
                a_used = x

            # y
            y = F.conv2d(
                a_used, w_used, bias=self.bias, 
                stride=self.stride, padding=self.padding, 
                dilation=self.dilation, groups=self.groups)
        else:   # if not self.training
            # if required, self.weight (w/ or w/o self.bias) has been already modified via self.quant_w()
            # else, self.weight (w/ or w/o self.bias) is used w/o any modification
            if self.qb_a < 32:
                if self.scale_a.nelement() == 0:
                    _, _, _, self.scale_a = LSQ.boot(self.qb_a, x, 'a', p=self.p_a)                    
                a_used = LSQ.quant(x, self.scale_a, self.n_a, self.p_a)
            else:
                a_used = x

            # y0
            y = F.conv2d(
                a_used, self.weight, bias=None, 
                stride=self.stride, padding=self.padding, 
                dilation=self.dilation, groups=self.groups)

            ## y0 --> y
            y = y * self.scale_w

            if self.qb_a < 32:
                y = y * self.scale_a

            if self.bias is not None:
                y = y + self.bias.view(1, -1, 1, 1)
            ## y0 --> y, end

        return y

    def quant_w_lsq(self):
        w_bar = LSQ.quant(self.weight, self.scale_w, self.n_w, self.p_w)
        self.weight = nn.Parameter(w_bar, requires_grad=False)
    ## LSQ, end

    ## DoReFa-Net
    def forward_dorefa(self, x):
        if self.training:
            w_used = DoReFa.forward(self.weight, 'w', self.p_w)

            if self.qb_a < 32:
                a_used = DoReFa.forward(x, 'a', self.p_a)                
            else:
                a_used = x

            # y
            y = F.conv2d(
                a_used, w_used, bias=self.bias, 
                stride=self.stride, padding=self.padding, 
                dilation=self.dilation, groups=self.groups)
        else:   # if not self.training
            # if required, self.weight (w/ or w/o self.bias) has been already modified via self.quant_w()
            # else, self.weight (w/ or w/o self.bias) is used w/o any modification
            if self.qb_a < 32:
                a_used = DoReFa.quant(x, 'a', self.p_a)
            else:
                a_used = x

            # y0
            y = F.conv2d(
                a_used, self.weight, bias=None, 
                stride=self.stride, padding=self.padding, 
                dilation=self.dilation, groups=self.groups)

            ## y0 --> y
            if self.p_w == 1:
                y = y * self.avg_abs_w
            else:
                y = (2.0 * y / self.p_w) - F.conv2d(
                    a_used, torch.ones_like(self.weight), bias=None, 
                    stride=self.stride, padding=self.padding, 
                    dilation=self.dilation, groups=self.groups)

            if self.qb_a < 32:
                y = y / self.p_a

            if self.bias is not None:
                y = y + self.bias.view(1, -1, 1, 1)
            ## y0 --> y, end

        return y

    def quant_w_dorefa(self):
        self.avg_abs_w = torch.mean(torch.abs(self.weight)).item()
        w_bar = DoReFa.quant(self.weight, 'w', self.p_w)
        self.weight = nn.Parameter(w_bar, requires_grad=False)
    ## DoReFa-Net, end

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        track_running_stats):
        # Initialize class
        super(BatchNorm2d, self).__init__(
            num_features,
            track_running_stats=track_running_stats)

        # Initialize params
        ones_(self.weight)  # gamma
        zeros_(self.bias)   # beta

class BatchNorm2dOnlyBeta(nn.BatchNorm2d):  # BatchNorm2d optimizing bias (beta) only
    def __init__(
        self,
        num_features,
        track_running_stats):
        # Initialize class
        super(BatchNorm2dOnlyBeta, self).__init__(
            num_features,
            track_running_stats=track_running_stats)

        # Initialize param
        zeros_(self.bias)

        # Weight (gamma) is fixed to 1
        delattr(self, 'weight')
        self.register_buffer(
            'weight',
            torch.ones(num_features))

class Linear(nn.Linear):
    def __init__(
        self,
        quant_scheme,
        qb_w, qb_a,        
        in_features, 
        out_features, 
        init_w_method,
        bias,
        **kwargs):
        # Initialize class
        super(Linear, self).__init__(in_features, out_features, bias=bias)

        ## Initialize param(s)
        # Weight
        if init_w_method == 'normal':
            normal_(self.weight, 0, 0.02)
            if bias:
                zeros_(self.bias)
        elif init_w_method == 'kaiming_uniform':
            kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                uniform_(self.bias, -bound, bound)
        ## Initialize param(s), end

        ## Prepare QAT & quantization
        self.quant_scheme = quant_scheme
        self.qb_w, self.qb_a = qb_w, qb_a

        if quant_scheme is None:
            assert (qb_w == 32) and (qb_a == 32)
        else:
            assert qb_w < 32

        if quant_scheme == 'lsq':  
            assert 'prev_act' in kwargs     # 'relu' or 'relu6' or 'swish' or 'etc'

            self.n_w, self.p_w, self.scale_grad_w, self.scale_w = LSQ.boot(qb_w, self.weight, 'w')    

            if qb_a < 32:
                self.n_a, self.p_a, self.scale_grad_a, self.scale_a = LSQ.boot(qb_a, None, 'a', prev_act=kwargs['prev_act'])
        elif quant_scheme == 'dorefa':
            self.p_w = DoReFa.boot(qb_w)

            if qb_a < 32:
                self.p_a = DoReFa.boot(qb_a)
        else:
            assert quant_scheme is None
        ## Prepare QAT & quantization, end

    ## Aggregation
    def forward(self, x):
        if self.quant_scheme == 'lsq':
            return self.forward_lsq(x)
        elif self.quant_scheme == 'dorefa':
            return self.forward_dorefa(x)
        else:
            return super(Linear, self).forward(x)

    def quant_w(self):
        if self.quant_scheme == 'lsq':
            self.quant_w_lsq()
        elif self.quant_scheme == 'dorefa':
            self.quant_w_dorefa()
        else:
            pass
    ## Aggregation, end

    ## LSQ
    def forward_lsq(self, x):
        if self.training:
            w_used = LSQ.apply(self.weight, self.scale_w, self.scale_grad_w, self.n_w, self.p_w)

            if self.qb_a < 32:
                if self.scale_grad_a is None:
                    _, _, self.scale_grad_a, _ = LSQ.boot(self.qb_a, x, 'a', p=self.p_a)
                if self.scale_a.nelement() == 0:
                    _, _, _, self.scale_a = LSQ.boot(self.qb_a, x, 'a', p=self.p_a)
                a_used = LSQ.apply(x, self.scale_a, self.scale_grad_a, self.n_a, self.p_a)                
            else:
                a_used = x

            # y
            y = F.linear(a_used, w_used, bias=self.bias)
        else:   # if not self.training
            # if required, self.weight (w/ or w/o self.bias) has been already modified via self.quant_w()
            # else, self.weight (w/ or w/o self.bias) is used w/o any modification
            if self.qb_a < 32:
                if self.scale_a.nelement() == 0:
                    _, _, _, self.scale_a = LSQ.boot(self.qb_a, x, 'a', p=self.p_a)                    
                a_used = LSQ.quant(x, self.scale_a, self.n_a, self.p_a)
            else:
                a_used = x

            # y0
            y = F.linear(a_used, self.weight, bias=None)

            ## y0 --> y
            y = y * self.scale_w

            if self.qb_a < 32:
                y = y * self.scale_a

            if self.bias is not None:
                y = y + self.bias.view(1, -1)
            ## y0 --> y, end

        return y

    def quant_w_lsq(self):
        w_bar = LSQ.quant(self.weight, self.scale_w, self.n_w, self.p_w)
        self.weight = nn.Parameter(w_bar, requires_grad=False)
    ## LSQ, end

    ## DoReFa-Net
    def forward_dorefa(self, x):
        if self.training:
            w_used = DoReFa.forward(self.weight, 'w', self.p_w)

            if self.qb_a < 32:
                a_used = DoReFa.forward(x, 'a', self.p_a)                
            else:
                a_used = x

            # y
            y = F.linear(a_used, w_used, bias=self.bias)
        else:   # if not self.training
            # if required, self.weight (w/ or w/o self.bias) has been already modified via self.quant_w()
            # else, self.weight (w/ or w/o self.bias) is used w/o any modification
            if self.qb_a < 32:
                a_used = DoReFa.quant(x, 'a', self.p_a)
            else:
                a_used = x

            # y0
            y = F.linear(a_used, self.weight, bias=None)

            ## y0 --> y
            if self.p_w == 1:
                y = y * self.avg_abs_w
            else:
                y = (2.0 * y / self.p_w) - F.linear(a_used, torch.ones_like(self.weight), bias=None)

            if self.qb_a < 32:
                y = y / self.p_a

            if self.bias is not None:
                y = y + self.bias.view(1, -1)
            ## y0 --> y, end

        return y

    def quant_w_dorefa(self):
        self.avg_abs_w = torch.mean(torch.abs(self.weight)).item()
        w_bar = DoReFa.quant(self.weight, 'w', self.p_w)
        self.weight = nn.Parameter(w_bar, requires_grad=False)
    ## DoReFa-Net, end

class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        quant_scheme,           # From this line: for conv layer
        qb_w, qb_a,
        in_planes,
        out_planes,
        init_w_method,
        bias,
        bn_layer,               # From this line: for BN layer
        track_running_stats,
        activation_layer,       # For activation function layer
        kernel_size=3,          # From this line: for conv layer again
        stride=1,
        dilation=1,
        groups=1,
        **kwargs):
        padding = (kernel_size - 1) // 2 * dilation

        ## Manage keyword arguments related to QAT & quantization
        kwargs_conv = {}

        if quant_scheme == 'lsq':
            assert 'prev_act' in kwargs
            kwargs_conv['prev_act'] = kwargs['prev_act']
        elif quant_scheme == 'dorefa':
            pass
        else:
            assert quant_scheme is None
        ## Manage keyword arguments related to QAT & quantization, end

        # Initialize class
        super(ConvBNActivation, self).__init__(
            Conv2d(
                quant_scheme, qb_w, qb_a,
                in_planes, out_planes, kernel_size, init_w_method, bias,
                stride=stride, padding=padding, 
                dilation=dilation, groups=groups, **kwargs_conv),
            bn_layer(out_planes, track_running_stats),
            activation_layer(inplace=True))

        self.out_channels = out_planes
## Common, end

## FOMAML
class MAMLConvNet(nn.Module):
    inter_qb_tuples = 3   # 5 - 2 (except first and last layers)

    def get_bits(self):
        idx = 0
        while idx < len(self.inter_qb_tuple_list):
            yield self.inter_qb_tuple_list[idx]
            idx += 1

    def __init__(
        self,
        quant_scheme,
        quant_scheme_kwargs,
        qb_w_first, qb_a_first,
        inter_qb_tuple_list, 
        qb_w_last, qb_a_last,
        in_channels,
        hidden_channels,
        classifier_in_features,
        num_classes,
        track_running_stats=False):
        # Initialize class
        super(MAMLConvNet, self).__init__()

        ## Clarify first- and last-layer bitwidth
        if qb_w_first == 'same':
            qb_w_first = inter_qb_tuple_list[0][0]
        if qb_a_first == 'same':
            qb_a_first = inter_qb_tuple_list[0][1]
        if qb_w_last == 'same':
            qb_w_last = inter_qb_tuple_list[-1][0]
        if qb_a_last == 'same':
            qb_a_last = inter_qb_tuple_list[-1][1]

        full_precision = True
        for (qb_w, qb_a) in inter_qb_tuple_list:
            if (qb_w < 32) or (qb_a < 32):
                full_precision = False
                break

        if full_precision:
            qb_w_first, qb_a_first = 32, 32
            qb_w_last, qb_a_last = 32, 32
        ## Clarify first- and last-layer bitwidth, end

        ## Manage keyword arguments related to QAT & quantization
        kwargs_conv1, kwargs_conv2, kwargs_conv3, kwargs_conv4, kwargs_fc = {}, {}, {}, {}, {}

        if quant_scheme == 'lsq':
            kwargs_conv1 = {'prev_act': 'etc'}
            kwargs_conv2 = {'prev_act': 'relu'}            
            kwargs_conv3 = {'prev_act': 'relu'} 
            kwargs_conv4 = {'prev_act': 'relu'}                     
            kwargs_fc = {'prev_act': 'relu'}
        elif quant_scheme == 'dorefa':
            pass
        else:
            assert quant_scheme is None   
        ## Manage keyword arguments related to QAT & quantization, end

        activation_layer = nn.ReLU
        self.qb_w_first, self.qb_a_first = qb_w_first, qb_a_first
        self.inter_qb_tuple_list = inter_qb_tuple_list
        self.qb_w_last, self.qb_a_last = qb_w_last, qb_a_last

        self.features1 = ConvBNActivation(
            None if (qb_w_first == 32) and (qb_a_first == 32) else quant_scheme, 
            qb_w_first, qb_a_first,
            in_channels, hidden_channels, 'trunc_normal', True,
            BatchNorm2dOnlyBeta, track_running_stats, activation_layer, **kwargs_conv1)

        qb_w, qb_a = self.get_bits().__next__()
        self.features2 = ConvBNActivation(
            None if (qb_w == 32) and (qb_a == 32) else quant_scheme,
            qb_w, qb_a,
            hidden_channels, hidden_channels, 'trunc_normal', True,
            BatchNorm2dOnlyBeta, track_running_stats, activation_layer, **kwargs_conv2)        

        qb_w, qb_a = self.get_bits().__next__()
        self.features3 = ConvBNActivation(
            None if (qb_w == 32) and (qb_a == 32) else quant_scheme,
            qb_w, qb_a,
            hidden_channels, hidden_channels, 'trunc_normal', True,
            BatchNorm2dOnlyBeta, track_running_stats, activation_layer, **kwargs_conv3)  

        qb_w, qb_a = self.get_bits().__next__()
        self.features4 = ConvBNActivation(
            None if (qb_w == 32) and (qb_a == 32) else quant_scheme,
            qb_w, qb_a,
            hidden_channels, hidden_channels, 'trunc_normal', True,
            BatchNorm2dOnlyBeta, track_running_stats, activation_layer, **kwargs_conv4)

        self.classifier = Linear(
            None if (qb_w_last == 32) and (qb_a_last == 32) else quant_scheme, 
            qb_w_last, qb_a_last, 
            classifier_in_features, num_classes, 'normal', True, **kwargs_fc)

        ## Check consistency
        self.conv_fc_layers = 0
        for m in self.modules():
            if type(m) in [Conv2d, Linear]:
                self.conv_fc_layers += 1

        if self.conv_fc_layers != len(self.inter_qb_tuple_list) + 2:
            raise Exception(
                f'2 measurements on # of conv/FC layers do not match: {self.conv_fc_layers} != {len(self.inter_qb_tuple_list)} + 2')
        ## Check consistency, end

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

    def quant_w(self):
        for m in self.modules():
            if type(m) in [Conv2d, Linear]:
                m.quant_w()

    def get_conv_fc_quant_cnt_lists(self):
        conv_fc_quant_cnt_list_w = [0] * (MAMLConvNet.inter_qb_tuples + 2)
        conv_fc_quant_cnt_list_a = [0] * (MAMLConvNet.inter_qb_tuples + 2)

        # NOTE: all modules MUST be registered in __init__() in the same order as forward()
        position = 0
        for key, m in self.named_modules():
            if type(m) in [Conv2d, Linear]:
                if m.qb_w < 32:
                    conv_fc_quant_cnt_list_w[position] += 1
                if m.qb_a < 32:
                    conv_fc_quant_cnt_list_a[position] += 1
                position += 1

        assert position == (MAMLConvNet.inter_qb_tuples + 2)

        return conv_fc_quant_cnt_list_w, conv_fc_quant_cnt_list_a 

    def get_conv_fc_info_dict(self):
        conv_fc_info_dict = {}  # Key: module name (key), value: order number

        # NOTE: all modules MUST be registered in __init__() in the same order as forward()
        position = 0
        for key, m in self.named_modules():
            if type(m) in [Conv2d, Linear]:
                conv_fc_info_dict[key] = position
                position += 1

        assert position == (MAMLConvNet.inter_qb_tuples + 2)

        return conv_fc_info_dict
## FOMAML, end

## Prototypical Networks
class ProtoConvNet(nn.Module):
    inter_qb_tuples = 3   # 4 - 1 (except first layer)

    def get_bits(self):
        idx = 0
        while idx < len(self.inter_qb_tuple_list):
            yield self.inter_qb_tuple_list[idx]
            idx += 1

    def __init__(
        self,
        quant_scheme,
        quant_scheme_kwargs,
        qb_w_first, qb_a_first,
        inter_qb_tuple_list, 
        in_channels,
        hidden_channels=64,
        out_channels=64,
        track_running_stats=True):
        # Initialize class
        super(ProtoConvNet, self).__init__()

        ## Clarify first- and last-layer bitwidth
        if qb_w_first == 'same':
            qb_w_first = inter_qb_tuple_list[0][0]
        if qb_a_first == 'same':
            qb_a_first = inter_qb_tuple_list[0][1]

        full_precision = True
        for (qb_w, qb_a) in inter_qb_tuple_list:
            if (qb_w < 32) or (qb_a < 32):
                full_precision = False
                break

        if full_precision:
            qb_w_first, qb_a_first = 32, 32
        ## Clarify first- and last-layer bitwidth, end

        ## Manage keyword arguments related to QAT & quantization
        kwargs_conv1, kwargs_conv2, kwargs_conv3, kwargs_conv4 = {}, {}, {}, {}

        if quant_scheme == 'lsq':
            kwargs_conv1 = {'prev_act': 'etc'}
            kwargs_conv2 = {'prev_act': 'relu'}            
            kwargs_conv3 = {'prev_act': 'relu'} 
            kwargs_conv4 = {'prev_act': 'relu'}                     
        elif quant_scheme == 'dorefa':
            pass
        else:
            assert quant_scheme is None   
        ## Manage keyword arguments related to QAT & quantization, end

        activation_layer = nn.ReLU
        self.qb_w_first, self.qb_a_first = qb_w_first, qb_a_first
        self.inter_qb_tuple_list = inter_qb_tuple_list

        self.features1 = ConvBNActivation(
            None if (qb_w_first == 32) and (qb_a_first == 32) else quant_scheme, 
            qb_w_first, qb_a_first,
            in_channels, hidden_channels, 'trunc_normal', True,
            BatchNorm2d, track_running_stats, activation_layer, **kwargs_conv1)

        qb_w, qb_a = self.get_bits().__next__()
        self.features2 = ConvBNActivation(
            None if (qb_w == 32) and (qb_a == 32) else quant_scheme,
            qb_w, qb_a,
            hidden_channels, hidden_channels, 'trunc_normal', True,
            BatchNorm2d, track_running_stats, activation_layer, **kwargs_conv2)        

        qb_w, qb_a = self.get_bits().__next__()
        self.features3 = ConvBNActivation(
            None if (qb_w == 32) and (qb_a == 32) else quant_scheme,
            qb_w, qb_a,
            hidden_channels, hidden_channels, 'trunc_normal', True,
            BatchNorm2d, track_running_stats, activation_layer, **kwargs_conv3)  

        qb_w, qb_a = self.get_bits().__next__()
        self.features4 = ConvBNActivation(
            None if (qb_w == 32) and (qb_a == 32) else quant_scheme,
            qb_w, qb_a,
            hidden_channels, out_channels, 'trunc_normal', True,
            BatchNorm2d, track_running_stats, activation_layer, **kwargs_conv4)

        ## Check consistency
        self.conv_layers = 0
        for m in self.modules():
            if type(m) == Conv2d:
                self.conv_layers += 1

        if self.conv_layers != len(self.inter_qb_tuple_list) + 1:
            raise Exception(
                f'2 measurements on # of conv layers do not match: {self.conv_layers} != {len(self.inter_qb_tuple_list)} + 1')
        ## Check consistency, end

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

    def quant_w(self):
        for m in self.modules():
            if type(m) == Conv2d:
                m.quant_w()

    def get_conv_quant_cnt_lists(self):
        conv_quant_cnt_list_w = [0] * (ProtoConvNet.inter_qb_tuples + 1)
        conv_quant_cnt_list_a = [0] * (ProtoConvNet.inter_qb_tuples + 1)

        # NOTE: all modules MUST be registered in __init__() in the same order as forward()
        position = 0
        for key, m in self.named_modules():
            if type(m) == Conv2d:
                if m.qb_w < 32:
                    conv_quant_cnt_list_w[position] += 1
                if m.qb_a < 32:
                    conv_quant_cnt_list_a[position] += 1
                position += 1

        assert position == (ProtoConvNet.inter_qb_tuples + 1)

        return conv_quant_cnt_list_w, conv_quant_cnt_list_a 

    def get_conv_info_dict(self):
        conv_info_dict = {}  # Key: module name (key), value: order number

        # NOTE: all modules MUST be registered in __init__() in the same order as forward()
        position = 0
        for key, m in self.named_modules():
            if type(m) == Conv2d:
                conv_info_dict[key] = position
                position += 1

        assert position == (ProtoConvNet.inter_qb_tuples + 1)

        return conv_info_dict
## Prototypical Networks, end

## Traditional training
def conv1x1(
    quant_scheme,
    qb_w, qb_a,    
    in_planes, 
    out_planes, 
    stride=1,
    **kwargs):
    return Conv2d(
        quant_scheme,
        qb_w, qb_a,
        in_planes, out_planes, 1, 'trunc_normal', False, stride=stride, **kwargs)

def conv3x3(
    quant_scheme,
    qb_w, qb_a,     
    in_planes, 
    out_planes, 
    stride=1, 
    groups=1, 
    dilation=1,
    **kwargs):
    return Conv2d(
        quant_scheme,
        qb_w, qb_a,
        in_planes, out_planes, 3, 'trunc_normal', False, 
        stride=stride, padding=dilation, groups=groups, dilation=dilation, **kwargs)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock20(nn.Module):
    expansion = 1

    def __init__(
        self, 
        quant_scheme,
        quant_scheme_kwargs,
        qb_tuple_list,        
        in_planes, 
        planes, 
        stride=1,
        track_running_stats=True,
        **kwargs):
        super(BasicBlock20, self).__init__()

        ## Manage keyword arguments related to QAT & quantization
        kwargs_conv1, kwargs_conv2 = {}, {}

        if quant_scheme == 'lsq':
            kwargs_conv1 = kwargs
            kwargs_conv2 = {'prev_act': 'relu'}
        elif quant_scheme == 'dorefa':
            pass
        else:
            assert quant_scheme is None   
        ## Manage keyword arguments related to QAT & quantization, end

        qb_w, qb_a = qb_tuple_list[0]
        self.conv1 = Conv2d(
            None if (qb_w == 32) and (qb_a == 32) else quant_scheme,
            qb_w, qb_a,            
            in_planes, planes, 3, 'trunc_normal', False, stride=stride, padding=1,
            **kwargs_conv1)
        self.bn1 = BatchNorm2d(planes, track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        qb_w, qb_a = qb_tuple_list[1]        
        self.conv2 = Conv2d(
            None if (qb_w == 32) and (qb_a == 32) else quant_scheme,
            qb_w, qb_a,             
            planes, planes, 3, 'trunc_normal', False, stride=1, padding=1,
            **kwargs_conv2)
        self.bn2 = BatchNorm2d(planes, track_running_stats)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        y += self.shortcut(x)
        y = self.relu(y)

        return y

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        quant_scheme,
        quant_scheme_kwargs,
        qb_tuple_list,
        in_planes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        track_running_stats=True,
        **kwargs):
        super(BasicBlock, self).__init__()

        ## Manage keyword arguments related to QAT & quantization
        kwargs_conv1, kwargs_conv2 = {}, {}

        if quant_scheme == 'lsq':
            kwargs_conv1 = kwargs
            kwargs_conv2 = {'prev_act': 'relu'}
        elif quant_scheme == 'dorefa':
            pass
        else:
            assert quant_scheme is None   
        ## Manage keyword arguments related to QAT & quantization, end

        if norm_layer is None:
            norm_layer = BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        qb_w, qb_a = qb_tuple_list[0]
        self.conv1 = conv3x3(
            None if (qb_w == 32) and (qb_a == 32) else quant_scheme,
            qb_w, qb_a,
            in_planes, planes,
            stride=stride, 
            **kwargs_conv1)
        self.bn1 = norm_layer(planes, track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        qb_w, qb_a = qb_tuple_list[1]
        self.conv2 = conv3x3(
            None if (qb_w == 32) and (qb_a == 32) else quant_scheme,
            qb_w, qb_a,
            planes, planes,
            **kwargs_conv2)
        self.bn2 = norm_layer(planes, track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if self.downsample is not None:
            identity = self.downsample(x)

        y += identity
        y = self.relu(y)

        return y

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        quant_scheme,
        quant_scheme_kwargs,
        qb_tuple_list,
        in_planes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        track_running_stats=True,
        **kwargs):
        super(Bottleneck, self).__init__()

        ## Manage keyword arguments related to QAT & quantization
        kwargs_conv1, kwargs_conv2, kwargs_conv3 = {}, {}, {}

        if quant_scheme == 'lsq':
            kwargs_conv1 = kwargs
            kwargs_conv2 = {'prev_act': 'relu'}
            kwargs_conv3 = {'prev_act': 'relu'}            
        elif quant_scheme == 'dorefa':
            pass
        else:
            assert quant_scheme is None   
        ## Manage keyword arguments related to QAT & quantization, end

        if norm_layer is None:
            norm_layer = BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        qb_w, qb_a = qb_tuple_list[0]
        self.conv1 = conv1x1(
            None if (qb_w == 32) and (qb_a == 32) else quant_scheme,
            qb_w, qb_a,
            in_planes, width,
            **kwargs_conv1)
        self.bn1 = norm_layer(width, track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        qb_w, qb_a = qb_tuple_list[1]
        self.conv2 = conv3x3(
            None if (qb_w == 32) and (qb_a == 32) else quant_scheme,
            qb_w, qb_a,            
            width, width, 
            stride=stride, groups=groups, dilation=dilation,
            **kwargs_conv2)
        self.bn2 = norm_layer(width, track_running_stats)
        qb_w, qb_a = qb_tuple_list[2]
        self.conv3 = conv1x1(
            None if (qb_w == 32) and (qb_a == 32) else quant_scheme,
            qb_w, qb_a,
            width, planes * self.expansion,
            **kwargs_conv3)
        self.bn3 = norm_layer(planes * self.expansion, track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)

        if self.downsample is not None:
            identity = self.downsample(x)

        y += identity
        y = self.relu(y)

        return y

class ResNet18(nn.Module):
    inter_qb_tuples = 19   # 21 - 2 (except first and last layers)

    def get_bits(self):
        idx = 0
        while idx < len(self.inter_qb_tuple_list):
            yield self.inter_qb_tuple_list[idx]
            idx += 1

    def __init__(
        self,
        quant_scheme,
        quant_scheme_kwargs,
        qb_w_first, qb_a_first,
        inter_qb_tuple_list, 
        qb_w_last, qb_a_last,
        num_classes,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        track_running_stats=True):
        super(ResNet18, self).__init__()

        # ResNet-18
        block = BasicBlock
        layers = [2, 2, 2, 2]

        ## Clarify first- and last-layer bitwidth
        if qb_w_first == 'same':
            qb_w_first = inter_qb_tuple_list[0][0]
        if qb_a_first == 'same':
            qb_a_first = inter_qb_tuple_list[0][1]
        if qb_w_last == 'same':
            qb_w_last = inter_qb_tuple_list[-1][0]
        if qb_a_last == 'same':
            qb_a_last = inter_qb_tuple_list[-1][1]

        full_precision = True
        for (qb_w, qb_a) in inter_qb_tuple_list:
            if (qb_w < 32) or (qb_a < 32):
                full_precision = False
                break

        if full_precision:
            qb_w_first, qb_a_first = 32, 32
            qb_w_last, qb_a_last = 32, 32
        ## Clarify first- and last-layer bitwidth, end

        ## Manage keyword arguments related to QAT & quantization
        kwargs_conv1, kwargs_conv_remaining, kwargs_fc = {}, {}, {}

        if quant_scheme == 'lsq':
            kwargs_conv1 = {'prev_act': 'etc'}
            kwargs_conv_remaining = {'prev_act': 'relu'}
            kwargs_fc = {'prev_act': 'relu'}
        elif quant_scheme == 'dorefa':
            pass
        else:
            assert quant_scheme is None

        self.kwargs_conv_remaining = kwargs_conv_remaining
        ## Manage keyword arguments related to QAT & quantization, end

        if norm_layer is None:
            norm_layer = BatchNorm2d
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.track_running_stats = track_running_stats
        self.inter_qb_tuple_list = inter_qb_tuple_list
        self.quant_scheme = quant_scheme
        self.quant_scheme_kwargs = quant_scheme_kwargs

        self.conv1 = Conv2d(
            None if (qb_w_first == 32) and (qb_a_first == 32) else quant_scheme,
            qb_w_first, qb_a_first,
            3, self.in_planes, 7, 'trunc_normal', False, stride=2, padding=3, **kwargs_conv1)
        self.bn1 = norm_layer(self.in_planes, track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(
            None if (qb_w_last == 32) and (qb_a_last == 32) else quant_scheme,
            qb_w_last, qb_a_last,
            512 * block.expansion, num_classes, 'normal', True, **kwargs_fc)

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
        if stride != 1 or self.in_planes != planes * block.expansion:
            qb_w, qb_a = self.get_bits().__next__()
            downsample = nn.Sequential(
                conv1x1(
                    None if (qb_w == 32) and (qb_a == 32) else self.quant_scheme,
                    qb_w, qb_a,
                    self.in_planes, planes * block.expansion,
                    stride=stride,
                    **self.kwargs_conv_remaining),
                norm_layer(planes * block.expansion, self.track_running_stats),
            )

        layers = []
        qb_w1, qb_a1 = self.get_bits().__next__()
        qb_w2, qb_a2 = self.get_bits().__next__()
        layers.append(
            block(
                self.quant_scheme,
                self.quant_scheme_kwargs,
                [(qb_w1, qb_a1), (qb_w2, qb_a2)],
                self.in_planes, planes, stride=stride, downsample=downsample, groups=self.groups, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer, 
                track_running_stats=self.track_running_stats, **self.kwargs_conv_remaining
            )
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            qb_w1, qb_a1 = self.get_bits().__next__()
            qb_w2, qb_a2 = self.get_bits().__next__()            
            layers.append(
                block(
                    self.quant_scheme,
                    self.quant_scheme_kwargs,
                    [(qb_w1, qb_a1), (qb_w2, qb_a2)],                  
                    self.in_planes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    track_running_stats=self.track_running_stats,
                    **self.kwargs_conv_remaining
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.fc(y)

        return y

    def quant_w(self):
        for m in self.modules():
            if type(m) in [Conv2d, Linear]:
                m.quant_w()

    def get_conv_fc_quant_cnt_lists(self):
        conv_fc_quant_cnt_list_w = [0] * (ResNet18.inter_qb_tuples + 2)
        conv_fc_quant_cnt_list_a = [0] * (ResNet18.inter_qb_tuples + 2)

        # NOTE: all modules MUST be registered in __init__() in the same order as forward()
        position = 0
        for key, m in self.named_modules():
            if type(m) in [Conv2d, Linear]:
                if m.qb_w < 32:
                    conv_fc_quant_cnt_list_w[position] += 1
                if m.qb_a < 32:
                    conv_fc_quant_cnt_list_a[position] += 1
                position += 1

        assert position == (ResNet18.inter_qb_tuples + 2)

        return conv_fc_quant_cnt_list_w, conv_fc_quant_cnt_list_a 

    def get_conv_fc_info_dict(self):
        conv_fc_info_dict = {}  # Key: module name (key), value: order number

        # NOTE: all modules MUST be registered in __init__() in the same order as forward()
        position = 0
        for key, m in self.named_modules():
            if type(m) in [Conv2d, Linear]:
                conv_fc_info_dict[key] = position
                position += 1

        assert position == (ResNet18.inter_qb_tuples + 2)

        return conv_fc_info_dict

class ResNet50(nn.Module):
    inter_qb_tuples = 52   # 54 - 2 (except first and last layers)

    def get_bits(self):
        idx = 0
        while idx < len(self.inter_qb_tuple_list):
            yield self.inter_qb_tuple_list[idx]
            idx += 1

    def __init__(
        self,
        quant_scheme,
        quant_scheme_kwargs,
        qb_w_first, qb_a_first,
        inter_qb_tuple_list, 
        qb_w_last, qb_a_last,
        num_classes,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        track_running_stats=True):
        super(ResNet50, self).__init__()

        # ResNet-50
        block = Bottleneck
        layers = [3, 4, 6, 3]

        ## Clarify first- and last-layer bitwidth
        if qb_w_first == 'same':
            qb_w_first = inter_qb_tuple_list[0][0]
        if qb_a_first == 'same':
            qb_a_first = inter_qb_tuple_list[0][1]
        if qb_w_last == 'same':
            qb_w_last = inter_qb_tuple_list[-1][0]
        if qb_a_last == 'same':
            qb_a_last = inter_qb_tuple_list[-1][1]

        full_precision = True
        for (qb_w, qb_a) in inter_qb_tuple_list:
            if (qb_w < 32) or (qb_a < 32):
                full_precision = False
                break

        if full_precision:
            qb_w_first, qb_a_first = 32, 32
            qb_w_last, qb_a_last = 32, 32
        ## Clarify first- and last-layer bitwidth, end

        ## Manage keyword arguments related to QAT & quantization
        kwargs_conv1, kwargs_conv_remaining, kwargs_fc = {}, {}, {}

        if quant_scheme == 'lsq':
            kwargs_conv1 = {'prev_act': 'etc'}
            kwargs_conv_remaining = {'prev_act': 'relu'}
            kwargs_fc = {'prev_act': 'relu'}
        elif quant_scheme == 'dorefa':
            pass
        else:
            assert quant_scheme is None

        self.kwargs_conv_remaining = kwargs_conv_remaining
        ## Manage keyword arguments related to QAT & quantization, end

        if norm_layer is None:
            norm_layer = BatchNorm2d
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.track_running_stats = track_running_stats
        self.inter_qb_tuple_list = inter_qb_tuple_list
        self.quant_scheme = quant_scheme
        self.quant_scheme_kwargs = quant_scheme_kwargs

        self.conv1 = Conv2d(
            None if (qb_w_first == 32) and (qb_a_first == 32) else quant_scheme,
            qb_w_first, qb_a_first,
            3, self.in_planes, 7, 'trunc_normal', False, stride=2, padding=3, **kwargs_conv1)
        self.bn1 = norm_layer(self.in_planes, track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(
            None if (qb_w_last == 32) and (qb_a_last == 32) else quant_scheme,
            qb_w_last, qb_a_last,
            512 * block.expansion, num_classes, 'normal', True, **kwargs_fc)

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
        if stride != 1 or self.in_planes != planes * block.expansion:
            qb_w, qb_a = self.get_bits().__next__()
            downsample = nn.Sequential(
                conv1x1(
                    None if (qb_w == 32) and (qb_a == 32) else self.quant_scheme,
                    qb_w, qb_a,
                    self.in_planes, planes * block.expansion,
                    stride=stride,
                    **self.kwargs_conv_remaining),
                norm_layer(planes * block.expansion, self.track_running_stats),
            )

        layers = []
        qb_w1, qb_a1 = self.get_bits().__next__()
        qb_w2, qb_a2 = self.get_bits().__next__()
        qb_w3, qb_a3 = self.get_bits().__next__()
        layers.append(
            block(
                self.quant_scheme,
                self.quant_scheme_kwargs,
                [(qb_w1, qb_a1), (qb_w2, qb_a2), (qb_w3, qb_a3)],
                self.in_planes, planes, stride=stride, downsample=downsample, groups=self.groups, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer, 
                track_running_stats=self.track_running_stats, **self.kwargs_conv_remaining
            )
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            qb_w1, qb_a1 = self.get_bits().__next__()
            qb_w2, qb_a2 = self.get_bits().__next__()
            qb_w3, qb_a3 = self.get_bits().__next__()            
            layers.append(
                block(
                    self.quant_scheme,
                    self.quant_scheme_kwargs,
                    [(qb_w1, qb_a1), (qb_w2, qb_a2), (qb_w3, qb_a3)],                  
                    self.in_planes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    track_running_stats=self.track_running_stats,
                    **self.kwargs_conv_remaining
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.fc(y)

        return y

    def quant_w(self):
        for m in self.modules():
            if type(m) in [Conv2d, Linear]:
                m.quant_w()

    def get_conv_fc_quant_cnt_lists(self):
        conv_fc_quant_cnt_list_w = [0] * (ResNet50.inter_qb_tuples + 2)
        conv_fc_quant_cnt_list_a = [0] * (ResNet50.inter_qb_tuples + 2)

        # NOTE: all modules MUST be registered in __init__() in the same order as forward()
        position = 0
        for key, m in self.named_modules():
            if type(m) in [Conv2d, Linear]:
                if m.qb_w < 32:
                    conv_fc_quant_cnt_list_w[position] += 1
                if m.qb_a < 32:
                    conv_fc_quant_cnt_list_a[position] += 1
                position += 1

        assert position == (ResNet50.inter_qb_tuples + 2)

        return conv_fc_quant_cnt_list_w, conv_fc_quant_cnt_list_a 

    def get_conv_fc_info_dict(self):
        conv_fc_info_dict = {}  # Key: module name (key), value: order number

        # NOTE: all modules MUST be registered in __init__() in the same order as forward()
        position = 0
        for key, m in self.named_modules():
            if type(m) in [Conv2d, Linear]:
                conv_fc_info_dict[key] = position
                position += 1

        assert position == (ResNet50.inter_qb_tuples + 2)

        return conv_fc_info_dict

class ResNet20(nn.Module):
    inter_qb_tuples = 18   # 20 - 2 (except first and last layers)

    def get_bits(self):
        idx = 0
        while idx < len(self.inter_qb_tuple_list):
            yield self.inter_qb_tuple_list[idx]
            idx += 1

    def __init__(
        self,
        quant_scheme,
        quant_scheme_kwargs,
        qb_w_first, qb_a_first,
        inter_qb_tuple_list, 
        qb_w_last, qb_a_last,
        num_classes,
        track_running_stats=True):
        super(ResNet20, self).__init__()

        # ResNet-20
        block = BasicBlock20
        layers = [3, 3, 3]

        ## Clarify first- and last-layer bitwidth
        if qb_w_first == 'same':
            qb_w_first = inter_qb_tuple_list[0][0]
        if qb_a_first == 'same':
            qb_a_first = inter_qb_tuple_list[0][1]
        if qb_w_last == 'same':
            qb_w_last = inter_qb_tuple_list[-1][0]
        if qb_a_last == 'same':
            qb_a_last = inter_qb_tuple_list[-1][1]

        full_precision = True
        for (qb_w, qb_a) in inter_qb_tuple_list:
            if (qb_w < 32) or (qb_a < 32):
                full_precision = False
                break

        if full_precision:
            qb_w_first, qb_a_first = 32, 32
            qb_w_last, qb_a_last = 32, 32
        ## Clarify first- and last-layer bitwidth, end

        ## Manage keyword arguments related to QAT & quantization
        kwargs_conv1, kwargs_conv_remaining, kwargs_fc = {}, {}, {}

        if quant_scheme == 'lsq':
            kwargs_conv1 = {'prev_act': 'etc'}
            kwargs_conv_remaining = {'prev_act': 'relu'}
            kwargs_fc = {'prev_act': 'relu'}
        elif quant_scheme == 'dorefa':
            pass
        else:
            assert quant_scheme is None

        self.kwargs_conv_remaining = kwargs_conv_remaining
        ## Manage keyword arguments related to QAT & quantization, end

        self.in_planes = 16

        self.track_running_stats = track_running_stats
        self.inter_qb_tuple_list = inter_qb_tuple_list
        self.quant_scheme = quant_scheme
        self.quant_scheme_kwargs = quant_scheme_kwargs

        self.conv1 = Conv2d(
            None if (qb_w_first == 32) and (qb_a_first == 32) else quant_scheme,
            qb_w_first, qb_a_first,            
            3, 16, 3, 'trunc_normal', False, stride=1, padding=1,
            **kwargs_conv1)
        self.bn1 = BatchNorm2d(16, track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.linear = Linear(
            None if (qb_w_last == 32) and (qb_a_last == 32) else quant_scheme,
            qb_w_last, qb_a_last,
            64, num_classes, 'normal', True, **kwargs_fc)

    def _make_layer(
        self, 
        block, 
        planes, 
        layers, 
        stride):
        strides = [stride] + [1] * (layers - 1)
        layers = []
        for stride in strides:
            qb_w1, qb_a1 = self.get_bits().__next__()
            qb_w2, qb_a2 = self.get_bits().__next__()
            layers.append(
                block(
                    self.quant_scheme,
                    self.quant_scheme_kwargs,
                    [(qb_w1, qb_a1), (qb_w2, qb_a2)],                    
                    self.in_planes, planes, stride=stride,
                    track_running_stats=self.track_running_stats, **self.kwargs_conv_remaining))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)

        y = F.avg_pool2d(y, y.size()[3])
        y = y.view(y.size(0), -1)
        y = self.linear(y)

        return y

    def quant_w(self):
        for m in self.modules():
            if type(m) in [Conv2d, Linear]:
                m.quant_w()

    def get_conv_fc_quant_cnt_lists(self):
        conv_fc_quant_cnt_list_w = [0] * (ResNet20.inter_qb_tuples + 2)
        conv_fc_quant_cnt_list_a = [0] * (ResNet20.inter_qb_tuples + 2)

        # NOTE: all modules MUST be registered in __init__() in the same order as forward()
        position = 0
        for key, m in self.named_modules():
            if type(m) in [Conv2d, Linear]:
                if m.qb_w < 32:
                    conv_fc_quant_cnt_list_w[position] += 1
                if m.qb_a < 32:
                    conv_fc_quant_cnt_list_a[position] += 1
                position += 1

        assert position == (ResNet20.inter_qb_tuples + 2)

        return conv_fc_quant_cnt_list_w, conv_fc_quant_cnt_list_a 

    def get_conv_fc_info_dict(self):
        conv_fc_info_dict = {}  # Key: module name (key), value: order number

        # NOTE: all modules MUST be registered in __init__() in the same order as forward()
        position = 0
        for key, m in self.named_modules():
            if type(m) in [Conv2d, Linear]:
                conv_fc_info_dict[key] = position
                position += 1

        assert position == (ResNet20.inter_qb_tuples + 2)

        return conv_fc_info_dict

def resnet(
    num_layers,
    pretrained,
    quant_scheme,
    quant_scheme_kwargs,
    qb_w_first, qb_a_first,
    inter_qb_tuple_list, 
    qb_w_last, qb_a_last,
    num_classes,
    **kwargs):
    if num_layers == 18:
        net = ResNet18(
            quant_scheme,
            quant_scheme_kwargs,
            qb_w_first, qb_a_first,
            inter_qb_tuple_list, 
            qb_w_last, qb_a_last,
            num_classes,
            **kwargs)

        if pretrained:
            net.load_state_dict(
                torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/resnet18-f37072fd.pth"),
                strict=False)
    elif num_layers == 50:
        net = ResNet50(
            quant_scheme,
            quant_scheme_kwargs,
            qb_w_first, qb_a_first,
            inter_qb_tuple_list, 
            qb_w_last, qb_a_last,
            num_classes,
            **kwargs)

        if pretrained:
            net.load_state_dict(
                torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/resnet50-0676ba61.pth"),
                strict=False)
    else:
        assert num_layers == 20

        net = ResNet20(
            quant_scheme,
            quant_scheme_kwargs,
            qb_w_first, qb_a_first,
            inter_qb_tuple_list, 
            qb_w_last, qb_a_last,
            num_classes,
            **kwargs)        

        if pretrained:
            checkpoint = torch.load('./resnet20-12fca82f.th')
            net.load_state_dict(
                checkpoint['state_dict'],
                strict=False)

    return net

class InvertedResidual(nn.Module):
    def __init__(
        self,
        quant_scheme,
        quant_scheme_kwargs,
        qb_tuple_list,
        inp,
        oup,
        stride,
        expand_ratio,
        bias,
        activation_layer,
        track_running_stats=True,
        **kwargs):
        # Initialize class
        super(InvertedResidual, self).__init__()

        ## Manage keyword arguments (related to QAT & quantization)
        kwargs_pw, kwargs_dw, kwargs_pw_linear = {}, {}, {}

        if quant_scheme == 'lsq':
            if expand_ratio != 1:
                kwargs_pw['prev_act'] = kwargs['prev_act']
                kwargs_dw['prev_act'] = 'relu' if activation_layer == nn.ReLU else 'relu6'
                kwargs_pw_linear['prev_act'] = 'relu' if activation_layer == nn.ReLU else 'relu6'
            else:
                kwargs_dw['prev_act'] = kwargs['prev_act']
                kwargs_pw_linear['prev_act'] = 'relu' if activation_layer == nn.ReLU else 'relu6'                
        elif quant_scheme == 'dorefa':
            pass
        else:
            assert quant_scheme is None
        ## Manage keyword arguments (related to QAT & quantization), end

        if expand_ratio != 1:
            qb_w_pw, qb_a_pw = qb_tuple_list[0]
            qb_w_dw, qb_a_dw = qb_tuple_list[1]
            qb_w_pw_linear, qb_a_pw_linear = qb_tuple_list[2]
        else:
            qb_w_dw, qb_a_dw = qb_tuple_list[0]
            qb_w_pw_linear, qb_a_pw_linear = qb_tuple_list[1]                        

        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []

        if expand_ratio != 1:
            layers.append(
                # pw
                ConvBNActivation(
                    None if (qb_w_pw == 32) and (qb_a_pw == 32) else quant_scheme, 
                    qb_w_pw, qb_a_pw,
                    inp, hidden_dim, 'trunc_normal', bias, BatchNorm2d, 
                    track_running_stats, activation_layer, 
                    kernel_size=1, **kwargs_pw))

        layers.append(
            # dw
            ConvBNActivation(
                None if (qb_w_dw == 32) and (qb_a_dw == 32) else quant_scheme, 
                qb_w_dw, qb_a_dw,
                hidden_dim, hidden_dim, 'trunc_normal', bias, BatchNorm2d,
                track_running_stats, activation_layer,
                stride=stride, groups=hidden_dim,
                **kwargs_dw))

        layers.extend([
            # pw-linear
            Conv2d(
                None if (qb_w_pw_linear == 32) and (qb_a_pw_linear == 32) else quant_scheme, 
                qb_w_pw_linear, qb_a_pw_linear,
                hidden_dim, oup, 1, 'trunc_normal', bias,
                **kwargs_pw_linear),
            BatchNorm2d(oup, track_running_stats)])

        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2ImageNet(nn.Module):
    inter_qb_tuples = 51   # 53 - 2 (except first and last layers)

    def get_bits(self):
        idx = 0
        while idx < len(self.inter_qb_tuple_list):
            yield self.inter_qb_tuple_list[idx]
            idx += 1

    def __init__(
        self,
        quant_scheme,
        quant_scheme_kwargs,
        qb_w_first, qb_a_first,
        inter_qb_tuple_list,
        qb_w_last, qb_a_last,
        num_classes,
        width_mult=1.0,
        inverted_residual_setting=None,
        round_nearest=8,
        track_running_stats=True):
        super(MobileNetV2ImageNet, self).__init__()

        activation_layer = nn.ReLU6

        ## Clarify first- and last-layer bitwidth
        if qb_w_first == 'same':
            qb_w_first = inter_qb_tuple_list[0][0]
        if qb_a_first == 'same':
            qb_a_first = inter_qb_tuple_list[0][1]
        if qb_w_last == 'same':
            qb_w_last = inter_qb_tuple_list[-1][0]
        if qb_a_last == 'same':
            qb_a_last = inter_qb_tuple_list[-1][1]

        full_precision = True
        for (qb_w, qb_a) in inter_qb_tuple_list:
            if (qb_w < 32) or (qb_a < 32):
                full_precision = False
                break

        if full_precision:
            qb_w_first, qb_a_first = 32, 32
            qb_w_last, qb_a_last = 32, 32
        ## Clarify first- and last-layer bitwidth, end

        ## Manage keyword arguments related to QAT & quantization
        kwargs_conv1, kwargs_conv2, kwargs_conv3, kwargs_fc = {}, {}, {}, {}

        if quant_scheme == 'lsq':
            kwargs_conv1 = {'prev_act': 'etc'}
            # kwargs_conv2['prev_act'] will be assigned later (on-demand)
            kwargs_conv2 = {}
            kwargs_conv3 = {'prev_act': 'etc'}  
            kwargs_fc = {'prev_act': 'relu6'}
        elif quant_scheme == 'dorefa':
            pass
        else:
            assert quant_scheme is None
        ## Manage keyword arguments related to QAT & quantization, end

        in_channels = 32
        last_channels = 1280

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
            raise ValueError("inverted_residual_setting should be non-empty "
                             f"or a 4-element list, got {inverted_residual_setting}")

        self.track_running_stats = track_running_stats
        self.inter_qb_tuple_list = inter_qb_tuple_list
        self.quant_scheme = quant_scheme
        self.quant_scheme_kwargs = quant_scheme_kwargs

        # building first layer
        in_channels = make_divisible(in_channels * width_mult, round_nearest)
        self.last_channels = make_divisible(last_channels * max(1.0, width_mult), round_nearest)
        features = [ConvBNActivation(
            None if (qb_w_first == 32) and (qb_a_first == 32) else quant_scheme,
            qb_w_first, qb_a_first,
            3, in_channels, 'trunc_normal', False, BatchNorm2d,
            track_running_stats, activation_layer,
            stride=2, **kwargs_conv1)]
        # building inverted residual blocks
        for k, (t, c, n, s) in enumerate(inverted_residual_setting):
            out_channels = make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                if k == 0 and i == 0:
                    if quant_scheme == 'lsq':
                        kwargs_conv2['prev_act'] = 'relu6'
                else:
                    kwargs_conv2['prev_act'] = 'etc'

                stride = s if i == 0 else 1

                if t != 1:
                    qb_w1, qb_a1 = self.get_bits().__next__()
                    qb_w2, qb_a2 = self.get_bits().__next__()
                    qb_w3, qb_a3 = self.get_bits().__next__()
                    qb_tuple_list = [(qb_w1, qb_a1), (qb_w2, qb_a2), (qb_w3, qb_a3)]
                else:
                    qb_w1, qb_a1 = self.get_bits().__next__()
                    qb_w2, qb_a2 = self.get_bits().__next__()  
                    qb_tuple_list = [(qb_w1, qb_a1), (qb_w2, qb_a2)]                  

                features.append(InvertedResidual(
                    quant_scheme,
                    quant_scheme_kwargs,
                    qb_tuple_list,
                    in_channels, out_channels, stride, t, False, activation_layer,
                    track_running_stats,
                    **kwargs_conv2))

                in_channels = out_channels
        # building last several layer
        qb_w, qb_a = self.get_bits().__next__()
        features.append(ConvBNActivation(
            None if (qb_w == 32) and (qb_a == 32) else quant_scheme, 
            qb_w, qb_a,
            in_channels, self.last_channels, 'trunc_normal', False, BatchNorm2d,
            track_running_stats, activation_layer,
            kernel_size=1, **kwargs_conv3))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier 
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            Linear(
                None if (qb_w_last == 32) and (qb_a_last == 32) else quant_scheme,
                qb_w_last, qb_a_last,
                self.last_channels, num_classes, 'normal', True,
                **kwargs_fc))

    def forward(self, x):
        y = self.features(x)
        y = F.avg_pool2d(y, 7)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)

        return y

    def quant_w(self):
        for m in self.modules():
            if type(m) in [Conv2d, Linear]:
                m.quant_w()

    def get_conv_fc_quant_cnt_lists(self):
        conv_fc_quant_cnt_list_w = [0] * (MobileNetV2ImageNet.inter_qb_tuples + 2)
        conv_fc_quant_cnt_list_a = [0] * (MobileNetV2ImageNet.inter_qb_tuples + 2)

        # NOTE: all modules MUST be registered in __init__() in the same order as forward()
        position = 0
        for key, m in self.named_modules():
            if type(m) in [Conv2d, Linear]:
                if m.qb_w < 32:
                    conv_fc_quant_cnt_list_w[position] += 1
                if m.qb_a < 32:
                    conv_fc_quant_cnt_list_a[position] += 1
                position += 1

        assert position == (MobileNetV2ImageNet.inter_qb_tuples + 2)

        return conv_fc_quant_cnt_list_w, conv_fc_quant_cnt_list_a 

    def get_conv_fc_info_dict(self):
        conv_fc_info_dict = {}  # Key: module name (key), value: order number

        # NOTE: all modules MUST be registered in __init__() in the same order as forward()
        position = 0
        for key, m in self.named_modules():
            if type(m) in [Conv2d, Linear]:
                conv_fc_info_dict[key] = position
                position += 1

        assert position == (MobileNetV2ImageNet.inter_qb_tuples + 2)

        return conv_fc_info_dict

class MobileNetV2CIFAR(nn.Module):
    inter_qb_tuples = 51   # 53 - 2 (except first and last layers)

    def get_bits(self):
        idx = 0
        while idx < len(self.inter_qb_tuple_list):
            yield self.inter_qb_tuple_list[idx]
            idx += 1

    def __init__(
        self,
        quant_scheme,
        quant_scheme_kwargs,
        qb_w_first, qb_a_first,
        inter_qb_tuple_list,
        qb_w_last, qb_a_last,
        num_classes,
        width_mult=1.0,
        inverted_residual_setting=None,
        round_nearest=8,
        track_running_stats=True):
        super(MobileNetV2CIFAR, self).__init__()

        activation_layer = nn.ReLU6

        ## Clarify first- and last-layer bitwidth
        if qb_w_first == 'same':
            qb_w_first = inter_qb_tuple_list[0][0]
        if qb_a_first == 'same':
            qb_a_first = inter_qb_tuple_list[0][1]
        if qb_w_last == 'same':
            qb_w_last = inter_qb_tuple_list[-1][0]
        if qb_a_last == 'same':
            qb_a_last = inter_qb_tuple_list[-1][1]

        full_precision = True
        for (qb_w, qb_a) in inter_qb_tuple_list:
            if (qb_w < 32) or (qb_a < 32):
                full_precision = False
                break

        if full_precision:
            qb_w_first, qb_a_first = 32, 32
            qb_w_last, qb_a_last = 32, 32
        ## Clarify first- and last-layer bitwidth, end

        ## Manage keyword arguments related to QAT & quantization
        kwargs_conv1, kwargs_conv2, kwargs_conv3, kwargs_fc = {}, {}, {}, {}

        if quant_scheme == 'lsq':
            kwargs_conv1 = {'prev_act': 'etc'}
            # kwargs_conv2['prev_act'] will be assigned later (on-demand)
            kwargs_conv2 = {}
            kwargs_conv3 = {'prev_act': 'etc'}  
            kwargs_fc = {'prev_act': 'relu6'}
        elif quant_scheme == 'dorefa':
            pass
        else:
            assert quant_scheme is None
        ## Manage keyword arguments related to QAT & quantization, end

        in_channels = 32
        last_channels = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             f"or a 4-element list, got {inverted_residual_setting}")

        self.track_running_stats = track_running_stats
        self.inter_qb_tuple_list = inter_qb_tuple_list
        self.quant_scheme = quant_scheme
        self.quant_scheme_kwargs = quant_scheme_kwargs

        # building first layer
        in_channels = make_divisible(in_channels * width_mult, round_nearest)
        self.last_channels = make_divisible(last_channels * max(1.0, width_mult), round_nearest)
        features = [ConvBNActivation(
            None if (qb_w_first == 32) and (qb_a_first == 32) else quant_scheme,
            qb_w_first, qb_a_first,
            3, in_channels, 'trunc_normal', False, BatchNorm2d,
            track_running_stats, activation_layer,
            stride=1, **kwargs_conv1)]  # NOTE: change conv1 stride 2 -> 1 for CIFAR
        # building inverted residual blocks
        for k, (t, c, n, s) in enumerate(inverted_residual_setting):
            out_channels = make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                if k == 0 and i == 0:
                    if quant_scheme == 'lsq':
                        kwargs_conv2['prev_act'] = 'relu6'
                else:
                    kwargs_conv2['prev_act'] = 'etc'

                stride = s if i == 0 else 1

                if t != 1:
                    qb_w1, qb_a1 = self.get_bits().__next__()
                    qb_w2, qb_a2 = self.get_bits().__next__()
                    qb_w3, qb_a3 = self.get_bits().__next__()
                    qb_tuple_list = [(qb_w1, qb_a1), (qb_w2, qb_a2), (qb_w3, qb_a3)]
                else:
                    qb_w1, qb_a1 = self.get_bits().__next__()
                    qb_w2, qb_a2 = self.get_bits().__next__()  
                    qb_tuple_list = [(qb_w1, qb_a1), (qb_w2, qb_a2)]                  

                features.append(InvertedResidual(
                    quant_scheme,
                    quant_scheme_kwargs,
                    qb_tuple_list,
                    in_channels, out_channels, stride, t, False, activation_layer,
                    track_running_stats,
                    **kwargs_conv2))

                in_channels = out_channels
        # building last several layer
        qb_w, qb_a = self.get_bits().__next__()
        features.append(ConvBNActivation(
            None if (qb_w == 32) and (qb_a == 32) else quant_scheme, 
            qb_w, qb_a,
            in_channels, self.last_channels, 'trunc_normal', False, BatchNorm2d,
            track_running_stats, activation_layer,
            kernel_size=1, **kwargs_conv3))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier 
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            Linear(
                None if (qb_w_last == 32) and (qb_a_last == 32) else quant_scheme,
                qb_w_last, qb_a_last,
                self.last_channels, num_classes, 'normal', True,
                **kwargs_fc))

    def forward(self, x):
        y = self.features(x)
        y = F.avg_pool2d(y, 4)  # NOTE: change pooling kernel_size 7 -> 4 for CIFAR
        y = y.view(y.size(0), -1)
        y = self.classifier(y)

        return y

    def quant_w(self):
        for m in self.modules():
            if type(m) in [Conv2d, Linear]:
                m.quant_w()

    def get_conv_fc_quant_cnt_lists(self):
        conv_fc_quant_cnt_list_w = [0] * (MobileNetV2CIFAR.inter_qb_tuples + 2)
        conv_fc_quant_cnt_list_a = [0] * (MobileNetV2CIFAR.inter_qb_tuples + 2)

        # NOTE: all modules MUST be registered in __init__() in the same order as forward()
        position = 0
        for key, m in self.named_modules():
            if type(m) in [Conv2d, Linear]:
                if m.qb_w < 32:
                    conv_fc_quant_cnt_list_w[position] += 1
                if m.qb_a < 32:
                    conv_fc_quant_cnt_list_a[position] += 1
                position += 1

        assert position == (MobileNetV2CIFAR.inter_qb_tuples + 2)

        return conv_fc_quant_cnt_list_w, conv_fc_quant_cnt_list_a 

    def get_conv_fc_info_dict(self):
        conv_fc_info_dict = {}  # Key: module name (key), value: order number

        # NOTE: all modules MUST be registered in __init__() in the same order as forward()
        position = 0
        for key, m in self.named_modules():
            if type(m) in [Conv2d, Linear]:
                conv_fc_info_dict[key] = position
                position += 1

        assert position == (MobileNetV2CIFAR.inter_qb_tuples + 2)

        return conv_fc_info_dict

def mobilenet_v2(
    dataset,
    pretrained,
    quant_scheme,
    quant_scheme_kwargs,
    qb_w_first, qb_a_first,
    inter_qb_tuple_list,
    qb_w_last, qb_a_last,
    num_classes,
    **kwargs):
    if dataset == 'imagenet':
        net = MobileNetV2ImageNet(
            quant_scheme,
            quant_scheme_kwargs,
            qb_w_first, qb_a_first,
            inter_qb_tuple_list,
            qb_w_last, qb_a_last,
            num_classes,
            **kwargs)

        if pretrained:
            net.load_state_dict(
                torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"),
                strict=False)   
    else:
        assert ('cifar' in dataset) and (not pretrained)

        net = MobileNetV2CIFAR(
            quant_scheme,
            quant_scheme_kwargs,
            qb_w_first, qb_a_first,
            inter_qb_tuple_list,
            qb_w_last, qb_a_last,
            num_classes,
            **kwargs)

    return net
## Traditional training, end

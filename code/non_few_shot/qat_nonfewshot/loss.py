import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
Written referring to:
    https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
    https://github.com/SHI-Labs/Any-Precision-DNNs
    https://github.com/JiahuiYu/slimmable_networks/
"""

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Loss function in [NeurIPS '17] Prototypical Networks for Few-shot Learning
def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def prototypical_loss(z, n_way, k_shot_sup, k_shot_qry):
    target_inds = torch.arange(0, n_way).view(n_way, 1, 1)
    target_inds = target_inds.expand(n_way, k_shot_qry, 1).long()
    target_inds = Variable(target_inds, requires_grad=False).to(device)

    z_dim = z.size(-1)

    z_proto = z[:(n_way*k_shot_sup)].view(n_way, k_shot_sup, z_dim).mean(1)
    zq = z[(n_way*k_shot_sup):]

    dists = euclidean_dist(zq, z_proto)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_way, k_shot_qry, -1)

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    _, y_hat = log_p_y.max(2)
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

    return loss_val, acc_val
##

# Loss function for knowledge distillation 
class CrossEntropyLossSoft(nn.modules.loss._Loss):
    def forward(self, output, target):
        output_log_prob = F.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        cross_entropy_loss = cross_entropy_loss.mean()
        return cross_entropy_loss

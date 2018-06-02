"""ops.py"""

import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def reconstruction_loss(x_recon, x, distribution):
    assert x_recon.size() == x.size()

    n = x.size(0)
    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(n)
    elif distribution == 'gaussian':
        x_recon = F.tanh(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(n)
    else:
        raise NotImplementedError('supported distributions: bernoulli | gaussian')

    return recon_loss


def mmd_imq(z_tilde, z, C):
    assert z_tilde.size() == z.size()
    assert z.ndimension() == 2

    n = z.size(0)
    out = sum_kernel_imq(z, z, C, exclude_diag=True).div(n*(n-1)) + \
          sum_kernel_imq(z_tilde, z_tilde, C, exclude_diag=True).div(n*(n-1)) - \
          sum_kernel_imq(z, z_tilde, C, exclude_diag=False).div(n*n).mul(2)

    return out


def sum_kernel_imq(z1, z2, C, exclude_diag=True):
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C/(1e-9+C+(z11-z22).pow(2).sum(2))
    if exclude_diag:
        eye = Variable(torch.diag(kernel_matrix.data.new(100).fill_(1)))
        kernel_matrix = kernel_matrix*(1-eye)

    kernel_sum = kernel_matrix.sum()
    return kernel_sum


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


def log_density_igaussian(z, z_var):
    """Calculate log density of zero-mean isotropic gaussian distribution given z and z_var."""
    assert z.ndimension() == 2
    assert z_var > 0

    z_dim = z.size(1)

    return -(z_dim/2)*math.log(2*math.pi*z_var) + z.pow(2).sum(1).div(-2*z_var)


def multistep_lr_decay(optimizer, current_step, schedules):
    """Manual LR scheduler for implementing schedules described in the WAE paper."""
    for step in schedules:
        if current_step == step:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']/schedules[step]

    return optimizer


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def kl_divergence(mu, logvar):
    assert mu.size() == logvar.size()
    assert mu.size(0) != 0

    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean()
    mean_kld = klds.mean()
    dimension_wise_kld = klds.mean(0)

    return total_kld, mean_kld, dimension_wise_kld

"""ops.py"""

import math

import torch.nn.functional as F


def reconstruction_loss(x_recon, x, distribution):
    r"""Calculate reconstruction loss for the general auto-encoder frameworks.

    Args:
        x_recon (Tensor): reconstructed images. arbitrary shape.
        x (Tensor): target images. same shape with x_recon.
        distribution (str): output distributions of the decoder. bernoulli or gaussian.
    """
    assert x_recon.size() == x.size()

    n = x.size(0)
    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(n)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(n)
    else:
        raise NotImplementedError('supported distributions: bernoulli/gaussian')

    return recon_loss


def mmd(z_tilde, z, z_var):
    r"""Calculate maximum mean discrepancy described in the WAE paper.

    Args:
        z_tilde (Tensor): samples from deterministic non-random encoder Q(Z|X).
            2D Tensor(batch_size x dimension).
        z (Tensor): samples from prior distributions. same shape with z_tilde.
        z_var (Number): scalar variance of isotropic gaussian prior P(Z).
    """
    assert z_tilde.size() == z.size()
    assert z.ndimension() == 2

    n = z.size(0)
    out = im_kernel_sum(z, z, z_var, exclude_diag=True).div(n*(n-1)) + \
          im_kernel_sum(z_tilde, z_tilde, z_var, exclude_diag=True).div(n*(n-1)) + \
          -im_kernel_sum(z, z_tilde, z_var, exclude_diag=False).div(n*n).mul(2)

    return out


def im_kernel_sum(z1, z2, z_var, exclude_diag=True):
    r"""Calculate sum of sample-wise measures of inverse multiquadratics kernel described in the WAE paper.

    Args:
        z1 (Tensor): batch of samples from a multivariate gaussian distribution \
            with scalar variance of z_var.
        z2 (Tensor): batch of samples from another multivariate gaussian distribution \
            with scalar variance of z_var.
        exclude_diag (bool): whether to exclude diagonal kernel measures before sum it all.
    """
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z_dim = z1.size(1)
    C = 2*z_dim*z_var

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C/(1e-9+C+(z11-z22).pow(2).sum(2))
    kernel_sum = kernel_matrix.sum()
    # numerically identical to the formulation. but..
    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()

    return kernel_sum


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


def squared_distance(tensor1, tensor2):
    assert tensor1.size() == tensor2.size()
    assert tensor1.ndimension() == 2

    return (tensor1-tensor2).pow(2).sum(1).mean()

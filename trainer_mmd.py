"""trainer_mmd.py"""

import os
import math
from tqdm import tqdm
import visdom

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from model import WAE, DisentangledWAE
from utils import DataGather, mkdirs
from ops import stochastic_panelty, reconstruction_loss, mmd, im_kernel_sum, multistep_lr_decay
from dataset import return_data


class Trainer(object):
    def __init__(self, args):
        self.name = args.name

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.max_epoch = args.max_epoch
        self.global_epoch = 0
        self.global_iter = 0
        self.iters_per_epoch = len(self.data_loader)
        self.max_iter = self.max_epoch*self.iters_per_epoch
        self.pbar = tqdm(total=self.max_iter)

        self.z_dim = args.z_dim # prior & posteiror dimension
        self.z_var = args.z_var # prior scalar variance
        self.z_sigma = math.sqrt(args.z_var) # sqrt of scalar variance
        self.Lambda = args.Lambda
        self.Lambda_p = args.Lambda_p
        self.lr_schedules = {30:2, 50:5, 100:10}

        if args.dataset.lower() == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == '3dchairs':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'dsprites':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        else:
            raise NotImplementedError

        #net = WAE
        net = DisentangledWAE
        self.net = net(self.z_dim, self.nc).to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr=args.lr,
                                    betas=(args.beta1, args.beta2))

        self.viz_on = args.viz_on
        self.win_id = dict(recon='win_recon', mmd='win_mmd', var='win_var')
        if self.viz_on:
            self.viz = visdom.Visdom(port=args.viz_port)
            self.line_gather = DataGather('iter', 'recon', 'mmd', 'var')
            self.image_gather = DataGather('true', 'recon')

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
        mkdirs(self.ckpt_dir)

        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.name)
        mkdirs(self.output_dir)


#    def train2(self):
#        self.net.train()
#        self.pbar.update(self.global_iter)
#
#        out = False
#        while not out:
#            for x in self.data_loader:
#                self.pbar.update(1)
#                self.global_iter += 1
#                if self.global_iter % iters_per_epoch == 0:
#                    self.global_epoch += 1
#                self.optim = multistep_lr_decay(self.optim, self.global_epoch, self.lr_schedules)
#
#                x = Variable(cuda(x, self.use_cuda))
#                x_recon, z_tilde = self.net(x)
#                z = self.sample_z(template=z_tilde, sigma=self.z_sigma)
#
#                recon_loss = F.mse_loss(x_recon, x, size_average=False).div(self.batch_size)
#                mmd_loss = mmd(z_tilde, z, z_var=self.z_var)
#                total_loss = recon_loss + self.Lambda*mmd_loss
#
#                self.optim.zero_grad()
#                total_loss.backward()
#                self.optim.step()
#
#                if self.global_iter%1000 == 0:
#                    self.line_gather.insert(iter=self.global_iter,
#                                    mu=z.mean(0).data, var=z.var(0).data,
#                                    recon_loss=recon_loss.data, mmd_loss=mmd_loss.data,)
#
#                if self.global_iter%5000 == 0:
#                    self.line_gather.insert(images=x.data)
#                    self.line_gather.insert(images=x_recon.data)
#                    self.viz_reconstruction()
#                    self.viz_lines()
#                    self.sample_x_from_z(n_sample=100)
#                    self.line_gather.flush()
#                    self.save_checkpoint('last')
#                    self.pbar.write('[{}] total_loss:{:.3f} recon_loss:{:.3f} mmd_loss:{:.3f}'.format(
#                        self.global_iter, total_loss.data[0], recon_loss.data[0], mmd_loss.data[0]))
#
#                if self.global_iter%20000 == 0:
#                    self.save_checkpoint(str(self.global_iter))
#
#
#                if self.global_iter >= max_iter:
#                    out = True
#                    break
#
#        pbar.write("[Training Finished]")

    def train(self):
        self.net.train()
        self.pbar.update(self.global_iter)

        out = False
        while not out:
            for x in self.data_loader:
                self.pbar.update(1)
                self.global_iter += 1
                if self.global_iter % self.iters_per_epoch == 0:
                    self.global_epoch += 1
                self.optim = multistep_lr_decay(self.optim, self.global_epoch, self.lr_schedules)

                x = x.to(self.device)
                x_recon, z_tilde, mu_tilde, logvar_tilde = self.net(x)
                z = self.sample_z(template=z_tilde, sigma=self.z_sigma)

                recon_loss = reconstruction_loss(x_recon, x, self.decoder_dist)
                mmd_loss = mmd(z_tilde, z, z_var=self.z_var)
                logvar_loss = stochastic_panelty(logvar_tilde)
                total_loss = recon_loss + self.Lambda*mmd_loss + self.Lambda_p*logvar_loss

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

#                if self.global_iter%1000 == 0:
#                    self.line_gather.insert(iter=self.global_iter,
#                                    mu=z.mean(0).data, var=z.var(0).data,
#                                    recon_loss=recon_loss.data, mmd_loss=mmd_loss.data,
#                                        logvar_loss=logvar_loss.data)

                if self.global_iter%500 == 0:
                    #self.line_gather.insert(images=x.data)
                    #self.line_gather.insert(images=x_recon.data)
                    #self.viz_reconstruction()
                    #self.viz_lines()
                    #self.sample_x_from_z(n_sample=100)
                    #self.line_gather.flush()
                    self.save_checkpoint('last')
                    self.pbar.write('[{}] total_loss:{:.3f} recon_loss:{:.3f} mmd_loss:{:.3f}'.format(
                        self.global_iter, total_loss.item(), recon_loss.item(), mmd_loss.item()))

                if self.global_iter%20000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")

    def viz_reconstruction(self):
        self.net.eval()
        x = self.line_gather.data['images'][0][:100]
        x = make_grid(x, normalize=True, nrow=10)
        x_recon = F.sigmoid(self.line_gather.data['images'][1][:100])
        x_recon = make_grid(x_recon, normalize=True, nrow=10)
        images = torch.stack([x, x_recon], dim=0).cpu()
        self.viz.images(images, env=self.name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=2)
        self.net.train()

    def viz_lines(self):
        self.net.eval()
        recon_losses = torch.stack(self.line_gather.data['recon_loss']).cpu()
        mmd_losses = torch.stack(self.line_gather.data['mmd_loss']).cpu()
        mus = torch.stack(self.line_gather.data['mu']).cpu()
        vars = torch.stack(self.line_gather.data['var']).cpu()
        iters = torch.Tensor(self.line_gather.data['iter'])

        legend = []
        for z_j in range(self.z_dim):
            legend.append('z_{}'.format(z_j))

        if self.win_recon is None:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        else:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.name+'_lines',
                                        win=self.win_recon,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))

        if self.win_mmd is None:
            self.win_mmd = self.viz.line(
                                        X=iters,
                                        Y=mmd_losses,
                                        env=self.name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='maximum mean discrepancy',))
        else:
            self.win_mmd = self.viz.line(
                                        X=iters,
                                        Y=mmd_losses,
                                        env=self.name+'_lines',
                                        win=self.win_mmd,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='maximum mean discrepancy',))

        if self.win_mu is None:
            self.win_mu = self.viz.line(
                                        X=iters,
                                        Y=mus,
                                        env=self.name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='posterior mean',))
        else:
            self.win_mu = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.name+'_lines',
                                        win=self.win_mu,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='posterior mean',))

        if self.win_var is None:
            self.win_var = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='posterior variance',))
        else:
            self.win_var = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.name+'_lines',
                                        win=self.win_var,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='posterior variance',))
        self.net.train()

    def sample_z(self, n_sample=None, dim=None, sigma=None, template=None):
        if n_sample is None:
            n_sample = self.batch_size
        if dim is None:
            dim = self.z_dim
        if sigma is None:
            sigma = self.z_sigma

        if template is not None:
            z = sigma*template.data.new(template.size()).normal_()
        else:
            z = sigma*torch.randn(n_sample, dim, device=self.device)

        return z

    def sample_x_from_z(self, n_sample):
        self.net.eval()
        z = self.sample_z(n_sample=n_sample, sigma=self.z_sigma)
        x_gen = F.sigmoid(self.net._decode(z)[:100]).data.cpu()
        x_gen = make_grid(x_gen, normalize=True, nrow=10)
        self.viz.images(x_gen, env=self.name+'_sampling_from_random_z',
                        opts=dict(title=str(self.global_iter)))
        self.net.train()

    def save_checkpoint(self, ckptname='last', verbose=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        states = {'iter':self.global_iter,
                  'epoch':self.global_epoch,
                  'model_states':model_states,
                  'optim_states':optim_states}

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            self.pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))

    def load_checkpoint(self, ckptname='last', verbose=True):
        if ckptname == 'ast':
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                if verbose:
                    self.pbar.write("=> no checkpoint found")
                return

            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.global_epoch = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            if verbose:
                self.pbar.write("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter))
        else:
            if verbose:
                self.pbar.write("=> no checkpoint found at '{}'".format(filepath))

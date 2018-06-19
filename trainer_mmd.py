"""trainer_mmd.py"""

import math
from pathlib import Path
from tqdm import tqdm
import visdom

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from model import WAE, DisentangledWAE
from utils import DataGather
from ops import stochastic_panelty, reconstruction_loss, mmd, im_kernel_sum, multistep_lr_decay, cuda
from dataset import return_data


class Trainer(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.max_epoch = args.max_epoch
        self.global_epoch = 0
        self.global_iter = 0

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
        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=args.lr,
                                    betas=(args.beta1, args.beta2))

        self.gather = DataGather()
        self.image_gather = DataGather()
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        if self.viz_on:
            self.viz = visdom.Visdom(env=self.viz_name+'_lines', port=self.viz_port)
            self.win_recon = None
            self.win_mmd = None
            self.win_mu = None
            self.win_var = None

        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.viz_name)
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = Path(args.output_dir).joinpath(args.viz_name)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

    def train2(self):
        self.net.train()

        iters_per_epoch = len(self.data_loader)
        max_iter = self.max_epoch*iters_per_epoch
        pbar = tqdm(total=max_iter)
        with tqdm(total=max_iter) as pbar:
            pbar.update(self.global_iter)
            out = False
            while not out:
                for x in self.data_loader:
                    pbar.update(1)
                    self.global_iter += 1
                    if self.global_iter % iters_per_epoch == 0:
                        self.global_epoch += 1
                    self.optim = multistep_lr_decay(self.optim, self.global_epoch, self.lr_schedules)

                    x = Variable(cuda(x, self.use_cuda))
                    x_recon, z_tilde = self.net(x)
                    z = self.sample_z(template=z_tilde, sigma=self.z_sigma)

                    recon_loss = F.mse_loss(x_recon, x, size_average=False).div(self.batch_size)
                    mmd_loss = mmd(z_tilde, z, z_var=self.z_var)
                    total_loss = recon_loss + self.Lambda*mmd_loss

                    self.optim.zero_grad()
                    total_loss.backward()
                    self.optim.step()

                    if self.global_iter%1000 == 0:
                        self.gather.insert(iter=self.global_iter,
                                        mu=z.mean(0).data, var=z.var(0).data,
                                        recon_loss=recon_loss.data, mmd_loss=mmd_loss.data,)

                    if self.global_iter%5000 == 0:
                        self.gather.insert(images=x.data)
                        self.gather.insert(images=x_recon.data)
                        self.viz_reconstruction()
                        self.viz_lines()
                        self.sample_x_from_z(n_sample=100)
                        self.gather.flush()
                        self.save_checkpoint('last')
                        pbar.write('[{}] total_loss:{:.3f} recon_loss:{:.3f} mmd_loss:{:.3f}'.format(
                            self.global_iter, total_loss.data[0], recon_loss.data[0], mmd_loss.data[0]))

                    if self.global_iter%20000 == 0:
                        self.save_checkpoint(str(self.global_iter))


                    if self.global_iter >= max_iter:
                        out = True
                        break

            pbar.write("[Training Finished]")

    def train(self):
        self.net.train()

        iters_per_epoch = len(self.data_loader)
        max_iter = self.max_epoch*iters_per_epoch
        pbar = tqdm(total=max_iter)
        with tqdm(total=max_iter) as pbar:
            pbar.update(self.global_iter)
            out = False
            while not out:
                for x in self.data_loader:
                    pbar.update(1)
                    self.global_iter += 1
                    if self.global_iter % iters_per_epoch == 0:
                        self.global_epoch += 1
                    self.optim = multistep_lr_decay(self.optim, self.global_epoch, self.lr_schedules)

                    x = Variable(cuda(x, self.use_cuda))
                    x_recon, z_tilde, mu_tilde, logvar_tilde = self.net(x)
                    z = self.sample_z(template=z_tilde, sigma=self.z_sigma)

                    recon_loss = reconstruction_loss(x_recon, x, self.decoder_dist)
                    mmd_loss = mmd(z_tilde, z, z_var=self.z_var)
                    logvar_loss = stochastic_panelty(logvar_tilde)
                    total_loss = recon_loss + self.Lambda*mmd_loss + self.Lambda_p*logvar_loss

                    self.optim.zero_grad()
                    total_loss.backward()
                    self.optim.step()

                    if self.global_iter%1000 == 0:
                        self.gather.insert(iter=self.global_iter,
                                        mu=z.mean(0).data, var=z.var(0).data,
                                        recon_loss=recon_loss.data, mmd_loss=mmd_loss.data,
                                           logvar_loss=logvar_loss.data)

                    if self.global_iter%5000 == 0:
                        self.gather.insert(images=x.data)
                        self.gather.insert(images=x_recon.data)
                        self.viz_reconstruction()
                        self.viz_lines()
                        self.sample_x_from_z(n_sample=100)
                        self.gather.flush()
                        self.save_checkpoint('last')
                        pbar.write('[{}] total_loss:{:.3f} recon_loss:{:.3f} mmd_loss:{:.3f}'.format(
                            self.global_iter, total_loss.item(), recon_loss.item(), mmd_loss.item()))

                    if self.global_iter%20000 == 0:
                        self.save_checkpoint(str(self.global_iter))

                    if self.global_iter >= max_iter:
                        out = True
                        break

            pbar.write("[Training Finished]")

    def viz_reconstruction(self):
        self.net.eval()
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True, nrow=10)
        x_recon = F.sigmoid(self.gather.data['images'][1][:100])
        x_recon = make_grid(x_recon, normalize=True, nrow=10)
        images = torch.stack([x, x_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=2)
        self.net.train()

    def viz_lines(self):
        self.net.eval()
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()
        mmd_losses = torch.stack(self.gather.data['mmd_loss']).cpu()
        mus = torch.stack(self.gather.data['mu']).cpu()
        vars = torch.stack(self.gather.data['var']).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        legend = []
        for z_j in range(self.z_dim):
            legend.append('z_{}'.format(z_j))

        if self.win_recon is None:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        else:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
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
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='maximum mean discrepancy',))
        else:
            self.win_mmd = self.viz.line(
                                        X=iters,
                                        Y=mmd_losses,
                                        env=self.viz_name+'_lines',
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
                                        env=self.viz_name+'_lines',
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
                                        env=self.viz_name+'_lines',
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
                                        env=self.viz_name+'_lines',
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
                                        env=self.viz_name+'_lines',
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
            z = sigma*Variable(template.data.new(template.size()).normal_())
        else:
            z = sigma*torch.randn(n_sample, dim)
            z = Variable(cuda(z, self.use_cuda))

        return z

    def sample_x_from_z(self, n_sample):
        self.net.eval()
        z = self.sample_z(n_sample=n_sample, sigma=self.z_sigma)
        x_gen = F.sigmoid(self.net._decode(z)[:100]).data.cpu()
        x_gen = make_grid(x_gen, normalize=True, nrow=10)
        self.viz.images(x_gen, env=self.viz_name+'_sampling_from_random_z',
                        opts=dict(title=str(self.global_iter)))
        self.net.train()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'mmd':self.win_mmd,
                      'mu':self.win_mu,
                      'var':self.win_var,}
        states = {'iter':self.global_iter,
                  'epoch':self.global_epoch,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(states, file_path.open('wb+'))
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename, silent=False):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            checkpoint = torch.load(file_path.open('rb'))
            self.global_iter = checkpoint['iter']
            self.global_epoch = checkpoint['epoch']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_mmd = checkpoint['win_states']['mmd']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            if not silent:
                print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            if not silent:
                print("=> no checkpoint found at '{}'".format(file_path))

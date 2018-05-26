"""trainer_gan.py"""

import math
from pathlib import Path
from tqdm import tqdm
import visdom

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from model import WAE, Adversary
from utils import DataGather
from ops import reconstruction_loss, mmd, im_kernel_sum, log_density_igaussian, multistep_lr_decay, cuda
from dataset import return_data


class Trainer(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_epoch = args.max_epoch
        self.global_epoch = 0
        self.global_iter = 0

        self.z_dim = args.z_dim
        self.z_var = args.z_var
        self.z_sigma = math.sqrt(args.z_var)
        self.prior_dist = torch.distributions.Normal(torch.zeros(self.z_dim),
                                                     torch.ones(self.z_dim)*self.z_sigma)
        self._lambda = args.reg_weight
        self.lr = args.lr
        self.lr_D = args.lr_D
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.lr_schedules = {30:2, 50:5, 100:10}

        if args.dataset.lower() == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        else:
            raise NotImplementedError

        self.net = cuda(WAE(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))

        self.D = cuda(Adversary(self.z_dim), self.use_cuda)
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                  betas=(self.beta1, self.beta2))

        self.gather = DataGather()
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        if self.viz_on:
            self.viz = visdom.Visdom(env=self.viz_name+'_lines', port=self.viz_port)
            self.win_recon = None
            self.win_QD = None
            self.win_D = None
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

    def train(self):
        self.net.train()

        ones = Variable(cuda(torch.ones(self.batch_size, 1), self.use_cuda))
        zeros = Variable(cuda(torch.zeros(self.batch_size, 1), self.use_cuda))

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
                    log_p_z = log_density_igaussian(z, self.z_var).view(-1, 1)

                    #D_z = self.D(z) + log_p_z.view(-1, 1)
                    #D_z_tilde = self.D(z_tilde) + log_p_z.view(-1, 1)
                    D_z = self.D(z)
                    D_z_tilde = self.D(z_tilde)
                    D_loss = F.binary_cross_entropy_with_logits(D_z+log_p_z, ones) + \
                             F.binary_cross_entropy_with_logits(D_z_tilde+log_p_z, zeros)
                    total_D_loss = self._lambda*D_loss

                    self.optim_D.zero_grad()
                    total_D_loss.backward(retain_graph=True)
                    self.optim_D.step()

                    recon_loss = F.mse_loss(x_recon, x, size_average=False).div(self.batch_size)
                    Q_loss = F.binary_cross_entropy_with_logits(D_z_tilde+log_p_z, ones)
                    total_AE_loss = recon_loss + self._lambda*Q_loss

                    self.optim.zero_grad()
                    total_AE_loss.backward()
                    self.optim.step()

                    if self.global_iter%10 == 0:
                        self.gather.insert(iter=self.global_iter,
                                           D_z=F.sigmoid(D_z).mean().detach().data,
                                           D_z_tilde=F.sigmoid(D_z_tilde).mean().detach().data,
                                           mu=z.mean(0).data,
                                           var=z.var(0).data,
                                           recon_loss=recon_loss.data,
                                           Q_loss=Q_loss.data,
                                           D_loss=D_loss.data)


                    if self.global_iter%50 == 0:
                        self.gather.insert(images=x.data)
                        self.gather.insert(images=x_recon.data)
                        self.viz_reconstruction()
                        self.viz_lines()
                        self.sample_x_from_z(n_sample=100)
                        self.gather.flush()
                        self.save_checkpoint('last')
                        pbar.write('[{}] recon_loss:{:.3f} Q_loss:{:.3f} D_loss:{:.3f}'.format(
                            self.global_iter, recon_loss.data[0], Q_loss.data[0], D_loss.data[0]))
                        pbar.write('D_z:{:.3f} D_z_tilde:{:.3f}'.format(
                            F.sigmoid(D_z).mean().detach().data[0],
                            F.sigmoid(D_z_tilde).mean().detach().data[0]))

                    if self.global_iter%2000 == 0:
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
        Q_losses = torch.stack(self.gather.data['Q_loss']).cpu()
        D_losses = torch.stack(self.gather.data['D_loss']).cpu()
        QD_losses = torch.cat([Q_losses, D_losses], 1)
        D_zs = torch.stack(self.gather.data['D_z']).cpu()
        D_z_tildes = torch.stack(self.gather.data['D_z_tilde']).cpu()
        Ds = torch.cat([D_zs, D_z_tildes], 1)
        mus = torch.stack(self.gather.data['mu']).cpu()
        vars = torch.stack(self.gather.data['var']).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        legend_z = []
        for z_j in range(self.z_dim):
            legend_z.append('z_{}'.format(z_j))

        legend_QD = ['Q_loss', 'D_loss']
        legend_D = ['D(z)', 'D(z_tilde)']

        if self.win_recon is None:
            self.win_recon = self.viz.line(X=iters,
                                           Y=recon_losses,
                                           env=self.viz_name+'_lines',
                                           opts=dict(
                                               width=400,
                                               height=400,
                                               xlabel='iteration',
                                               title='reconsturction loss',))
        else:
            self.win_recon = self.viz.line(X=iters,
                                           Y=recon_losses,
                                           env=self.viz_name+'_lines',
                                           win=self.win_recon,
                                           update='append',
                                           opts=dict(
                                               width=400,
                                               height=400,
                                               xlabel='iteration',
                                               title='reconsturction loss',))

        if self.win_QD is None:
            self.win_QD = self.viz.line(X=iters,
                                        Y=QD_losses,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend_QD,
                                            xlabel='iteration',
                                            title='Q&D Losses',))
        else:
            self.win_QD = self.viz.line(X=iters,
                                        Y=QD_losses,
                                        env=self.viz_name+'_lines',
                                        win=self.win_QD,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend_QD,
                                            xlabel='iteration',
                                            title='Q&D Losses',))

        if self.win_D is None:
            self.win_D = self.viz.line(X=iters,
                                       Y=Ds,
                                       env=self.viz_name+'_lines',
                                       opts=dict(
                                           width=400,
                                           height=400,
                                           legend=legend_D,
                                           xlabel='iteration',
                                           title='D(.)',))
        else:
            self.win_D = self.viz.line(X=iters,
                                       Y=Ds,
                                       env=self.viz_name+'_lines',
                                       win=self.win_D,
                                       update='append',
                                       opts=dict(
                                           width=400,
                                           height=400,
                                           legend=legend_D,
                                           xlabel='iteration',
                                           title='D(.)',))

        if self.win_mu is None:
            self.win_mu = self.viz.line(X=iters,
                                        Y=mus,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend_z,
                                            xlabel='iteration',
                                            title='posterior mean',))
        else:
            self.win_mu = self.viz.line(X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        win=self.win_mu,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend_z,
                                            xlabel='iteration',
                                            title='posterior mean',))

        if self.win_var is None:
            self.win_var = self.viz.line(X=iters,
                                         Y=vars,
                                         env=self.viz_name+'_lines',
                                         opts=dict(
                                             width=400,
                                             height=400,
                                             legend=legend_z,
                                             xlabel='iteration',
                                             title='posterior variance',))
        else:
            self.win_var = self.viz.line(X=iters,
                                         Y=vars,
                                         env=self.viz_name+'_lines',
                                         win=self.win_var,
                                         update='append',
                                         opts=dict(
                                             width=400,
                                             height=400,
                                             legend=legend_z,
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
        model_states = {'net':self.net.state_dict(),
                        'D':self.D.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),
                        'optim_D':self.optim_D.state_dict()}
        win_states = {'recon':self.win_recon,
                      'QD':self.win_QD,
                      'D':self.win_D,
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
            self.win_QD = checkpoint['win_states']['QD']
            self.win_D = checkpoint['win_states']['D']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            self.D.load_state_dict(checkpoint['model_states']['D'])
            self.optim_D.load_state_dict(checkpoint['optim_states']['optim_D'])
            if not silent:
                print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            if not silent:
                print("=> no checkpoint found at '{}'".format(file_path))

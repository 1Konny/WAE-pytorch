"""dataset.py"""

from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


def normalize_dsprites(img_tensor):
    """Normalize dspirtes-dataset to be in [-1, 1]"""
    return img_tensor*2-1


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        if transform is not None:
            self.data_tensor = transform(data_tensor)
        else:
            self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class CustomCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def add(self, transforms):
        self.transforms += transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    decoder_dist = args.decoder_dist

    if name.lower() == 'celeba':
        root = Path(dset_dir).joinpath('CelebA_trainval')
        transform = CustomCompose([
            transforms.CenterCrop((140, 140)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),])
        if decoder_dist == 'gaussian':
            transform.add([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder

    elif name.lower() == 'dsprites':
        root = Path(dset_dir).joinpath('dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not root.exists():
            import subprocess
            print('Now download dsprites-dataset')
            subprocess.call(['./download_dsprites.sh'])
            print('Download Finished')
        data = np.load(root, encoding='latin1')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        if decoder_dist == 'bernoulli':
            transform = None
        elif decoder_dist == 'gaussian':
            transform = normalize_dsprites
        train_kwargs = {'data_tensor':data, 'transform':transform}
        dset = CustomTensorDataset

    else:
        raise NotImplementedError

    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader
    return data_loader

if __name__ == '__main__':
    pass

# WAE-pytorch

### Dependencies
```
python 3.6.4
pytorch 0.3.1.post2
visdom
```
<br>

### Usage
1. download ```img_align_celeba.zip``` and ```list_eval_partition.txt``` files from [here], make ```data``` directory, put downloaded files into ```data```, and then run ```./preprocess_celeba.sh```. for example,
```
.
└── data
  └── img_align_celeba.zip
  └── list_eval_partition.txt
```
2. initialize visdom
```
python -m visdom.server
```
3. run by scripts
```
sh run_celeba_wae_mmd.sh
```
4. check training process on the visdom server
```
localhost:8097
```
<br>

## Results - CelebA
### train data reconstruction
![train_recon](misc/train_reconstruction.jpg)
### test data reconstruction
![test_recon](misc/test_reconstruction.jpg)
### random data generation via sampling z from P(z)
<p align="center">
<img src=misc/random_sample.jpg>
</p>
### training plots
![curves](misc/curves.png)

[here]: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

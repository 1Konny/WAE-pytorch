#! /bin/sh

python main.py --dataset CelebA --model mmd --max_epoch 250 --batch_size 100 \
    --lr 1e-3 --beta1 0.5 --beta2 0.999 --z_dim 64 --z_var 2 --reg_weight 100 \
    --viz_name wae_mmd_celeba

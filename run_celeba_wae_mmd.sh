#! /bin/sh

python main.py --dataset CelebA --model mmd --max_epoch 250 --batch_size 100 \
    --lr 1e-3 --beta1 0.5 --beta2 0.999 --z_dim 64 --z_var 2 --reg_weight 100 \
    --gather_step 200 --display_step 1000 --save_ckpt_step 2000 \
    --decoder_dist gaussian --viz_name wae_mmd_celeba

python3 pytorch-CycleGAN-and-pix2pix/train.py \
    --dataroot data/movie_style/star_wars_simple \
    --name SW_CycleGAN_reduced_v01 \
    --model cycle_gan \
    --load_size 512 \
    --crop_size 256 \
    --n_epochs 100

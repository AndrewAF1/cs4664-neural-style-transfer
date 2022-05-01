python3 pytorch-CycleGAN-and-pix2pix/train.py \
    --dataroot data/movie_style/star_wars_full \
    --name SW_CycleGAN_fulldataset_256pix \
    --model cycle_gan \
    --n_epochs 40 \
    --load_size 256 \
    --crop_size 256

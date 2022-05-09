python3 pytorch-CycleGAN-and-pix2pix/test.py \
    --dataroot data/movie_style/star_wars_full \
    --name SW_CycleGAN_fulldataset_256pix \
    --model cycle_gan \
    --preprocess 'scale_width' \
    --load_size 256

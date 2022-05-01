python3 pytorch-CycleGAN-and-pix2pix/train.py \
    --dataroot data/movie_style/star_wars_full \
    --name SW_CycleGAN_fulldataset_512pix \
    --model cycle_gan \
    --load_size 512 \
    --crop_size 256 \
    --continue_train

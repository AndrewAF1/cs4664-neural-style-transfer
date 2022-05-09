python3 pytorch-CycleGAN-and-pix2pix/train.py \
    --dataroot data/movie_style/star_wars_full \
    --name SW_CycleGAN_fulldataset_512pix_batch4 \
    --model cycle_gan \
    --load_size 512 \
    --crop_size 256 \
    --batch_size 4 \
    --continue_train

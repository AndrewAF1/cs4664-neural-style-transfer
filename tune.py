import os


batches = [2**n for n in range(7)]
for batch in batches:
    cmd = "python3 pytorch-CycleGAN-and-pix2pix/train.py --dataroot ./datasets/ncdataset --n_epochs 1 --batch_size {} --name ncd_pix2pix --model colorization".format(
        batch
    )

    os.system(cmd)


# script to test diff hyperparameters on grayscale images without pairs
import os

# batches = [2**n for n in range(7)]
# for batch in batches:
#     cmd = "python3 pytorch-CycleGAN-and-pix2pix/test.py --dataroot data/pix2pix/ncdataset --batch_size {} --name ncd_pix2pix_batch_{} --model colorization".format(
#         batch, batch
#     )

#     os.system(cmd)

epochs = [(2,8), (4, 6), (5, 5), (6, 4), (8, 2)]

for n_epochs, n_epochs_decay in epochs:
    cmd = "python3 pytorch-CycleGAN-and-pix2pix/test.py --dataroot data/pix2pix/ncdataset --name ncd_pix2pix_epochs_{}_{} --model colorization".format(
        n_epochs, n_epochs_decay
    )

    os.system(cmd)

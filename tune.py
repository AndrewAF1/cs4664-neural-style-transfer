# script to tune hyperparameters
import os

# batches = [2**n for n in range(7)]
# for batch in batches:
#     cmd = "python3 pytorch-CycleGAN-and-pix2pix/train.py --dataroot data/pix2pix/ncdataset --n_epochs 5 --n_epochs_decay 5 --batch_size {} --name ncd_pix2pix_batch_{} --model colorization".format(
#         batch, batch
#     )

#     os.system(cmd)

learning_rates = [0.00002, 0.0001, 0.0002, 0.0005]
for lr in learning_rates:
    cmd = "python3 pytorch-CycleGAN-and-pix2pix/train.py --dataroot data/pix2pix/ncdataset --n_epochs 5 --n_epochs_decay 5 --lr {} --name ncd_pix2pix_lr_{} --model colorization".format(
        lr, lr
    )

    os.system(cmd)


# epochs = [(2,8), (4, 6), (5, 5), (6, 4), (8, 2)]

# for n_epochs, n_epochs_decay in epochs:
#     cmd = "python3 pytorch-CycleGAN-and-pix2pix/train.py --dataroot data/pix2pix/ncdataset --n_epochs {} --n_epochs_decay {} --name ncd_pix2pix_epochs_{}_{} --model colorization".format(
#         n_epochs, n_epochs_decay,
#         n_epochs, n_epochs_decay
#     )

#     os.system(cmd)


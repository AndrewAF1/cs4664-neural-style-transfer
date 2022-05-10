Welcome! So you're interested computer vision-- where do you start? **Image-to-image translation** is a specific class of vision and graphics problems that can help introduce you into the field. First, let's define what image-to-image translation is: given images in some source domain A, the goal is to map the image into some target domain B while still retaining the content representations. It's analagous to how we might use technology like Google Translate to translate sentences from English to Spanish so that they have the same meaning. Effective deep learning models can do this translation much quicker and more efficiently than the average person. 

There are several applications of image-to-image translation due to its applicability to multiple domains.  Existing methods include those for 

- aerial photography to maps; 
- daytime to nighttime photography; 
- sketch outlines to colored images; 
- superresolution;
- etc. 

There are also a lot of different tools and techniques that can be used to implement these applications, the most common of which are **Generative Adversarial Networks (GANs)**. To give a basic explanation, GANs are a neural network that typically contain a generator model to create new synthetic images in the target domain and a discriminator model that returns a probability of how likely it is that the image is real or fake. By directly competing with one another, both models improve so that the generator is able to learn how to create more and more realistic images and the discriminator improves its ability to discern whether an image is real during testing. 

> **NOTE:** 
> 
> There are many different types of GANs that exist-- for example, **Pix2Pix** is a specific type of GAN that was designed for general image-to-image translation purposes. Pix2Pix uses **paired datasets** for training where there are images in the source domain that correspond directly to images in the target domain. 
> 
> **CycleGANs** are also a type of GAN, but differ in that they can use **unpaired datasets**-- aka datasets that have images in both domains that do not have matches in the other-- by using *two* generators and *two* discriminators. To train these models, the CycleGAN essentially generates an image from source domain A to target domain B, then generates a new image in domain A based on the image from domain B. One of the discriminators will then compare how this new image in domain A compares to the original, allowing the generators to improve. The CycleGAN will also take images from domain B, translate to domain A, then translate again to domain B and compare with another discriminator to ensure that both generators (and discriminators) are being trained properly to create realistic images. 

This repository contains code, documentation, dataset, and results for a Pix2Pix model that translates black-and-white images to color and a CycleGAN model that attempts to do a style transfer so that older movies are updated with modern day effects. The source code for both models is credited to [Jun-Yan Zhu](https://github.com/junyanz)  and [Taesung Park](https://github.com/taesungp) and was supported by [Tongzhou Wang](https://github.com/SsnL). Our team-- Andrew Farabow, Aditi Diwan, and Nivi Ramalingam-- updated their work with datasets and code that we used to run experiments on both models that can be used to further understand how GANs work. The results can also be found within the repository. 

## Table of Contents
1. [Getting Started](#getting-started)
2. [Black and White to Color](#black-and-white-to-color)
      * [Data Preprocessing](#data-pre-processing)
      * [Training](#training)
      * [Experiment 1: Running Open-Source Implementation with NCD](#experiment-1-running-open-source-implementation-with-ncd)
      * [Experiment 2: Hyperparameter Optimization](#experiment-2-hyperparameter-optimization)
      * [Experiment 3: Comparison of Generator/Discriminator Architectures](#experiment-3-comparison-of-generatordiscriminator-architectures)
      * [Experiment 4: Comparison with Baseline Model(s)](#experiment-4-comparison-with-baseline-models)
      * [Experiment 5: Assessment of Performance on Landscapes Dataset](#experiment-5-assessment-of-performance-on-landscapes-dataset)
3. [Special FX Style Transfer](#special-fx-style-transfer)
      * [Data Preprocessing](#data-pre-processing-1)
      * [Training](#training-1)
      * [Experiment 1: Running Open-Source Implementation](#experiment-1-running-open-source-implementation)
      * [Experiment 2: Higher Image Resolution](#experiment-2-higher-image-resolution)
      * [Experiment 3: Expanded Star Wars Dataset](#experiment-3-expanded-star-wars-dataset)
      * [Experiment 4: Batch size and Longer Training](#experiment-4-batch-size-and-longer-training)
5. [References](#references)
## Getting Started
Before working with the code, ensure that the following pre-requisites are met:
-  Linux or macOS
-  Python 3
-  CPU or NVIDIA GPU + CUDA CuDNN

To clone this repo (after ensuring that your SSH key is set up), use the commands
```
git clone git@github.com:AndrewAF1/cs4664-pix2pix.git
git submodule update --init
```
Install the required dependencies-- these can also be found in the file [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/003efc4c8819de47ff11b5a0af7ba09aee7f5fc1)/requirements.txt
- torch>=1.4.0
- torchvision>=0.5.0
- dominate>=2.4.0
- visdom>=0.1.8.8
- wandb

Now you are ready to begin working with your models!

## Black-and-White to Color
The Pix2Pix model that's available through this repository can handle multiple types of translation problems-- we'll specifically be looking at taking black-and-white images and colorizing them. We've provided a Natural Color Dataset (NCD) filled with paired images of different fruits and vegetables with over 700 images and the Landscapes dataset with over 7,000 images for training and testing. You're free to use your own data, but make sure that the data is organized into the correct structure so that the model will be able to access it. From there, train the model with the paired dataset. In the interest of understanding how this model works, we've also included a few experiments we ran and what the results meant to us. 

### Data Pre-Processing
If you do choose to use your own dataset, there will be some pre-processing in the form directory structures that need to be done. TODO 

### Training
Commands for training; for best results, train for...; talk about visdom server TODO

### Experiment 1: Running Open-Source Implementation with NCD
We recommend that you run this experiment as well--by running the implementation with NCD, we were able to gain insight into how the default model functions. Note that we had trouble running the model on our local computers, so we switched over to Virginia Tech's Advanced Research Computing (ARC) systems so that we could use the GPUs to speed up training time. We then had to restructure the NCD (see above) before running the model so that it would be able to find the images. Originally, we ran the ```combine.py``` to combine the grayscale and color pairs before feeding them into the model; we discovered later that although this step is necessary for the other translations, it caused color bleeding and other issues when used for the colorization model. 

### Experiment 2: Hyperparameter Optimization
The clear next step was to optimize this model so that it would be able to generate the most realistic looking color images. To do this, we needed to test out different values for some hyperparameters. We initially planned on trying out different combinations-- however, since the training times were so lengthy, we decided to use the ```tune.py``` script to run each model with a variety of options for an individual hyperparameter and combine the best performing hyperparameters at the end. In the interest of time, we also chose to focus on 4 hyperparameters: batch size, learning rates, number of epochs, and decay epoch. 

We had two runs of this experiment. In the initial run, we focused on all aforementioned hyper parameters. The goals for this run was to to test a large variety of values, so we set the total number of epochs to be 10 (`n_epochs` + `n_epochs_decay`). We kept this number small to ensure the training time was feasible. We found that batch size of 1 performed significantly better than the other values, so we used that as a default in the second run of this experiment. The ratio of number of epochs and number of decay epochs seemed to have little impact on the results, but it may have been due to the fact that we had a low overall number of epochs.  

In the second run of this experiment, we tried to have a more focused approach. We used the default learning rate as a baseline and chose value close to it and a value a bit further from it. We also increased the total number of epochs to 50 to allow our model to train gain more exposure to the dataset while still allowing us to train in a reasonable window of time. We found that the ratio of epochs still did not same to make much of a difference in performance of the model. 

We also have some results [here](/default_pix2pix_intermediate_results/index.html) demonstrating intermediate results of the default model with 50 epochs that we had looked at as part of this experiment.


### Experiment 3: Comparison of Generator/Discriminator Architectures
There are multiple different architectures that the generator and discriminator models can have. As an extension of the previous experiment, we tried training our model with the optimal hyper parameters by going down the list of possible options we could use for the generator architectures. We were able to successfully train the model, but came across issues when we tried to test the models. All of the except the default model threw errors, which we were unable to fix in the given time frame, but aim to fix in further experimentation.

### Experiment 4: Comparison with Baseline Model(s)
One of the drawbacks of using a GAN is that they typically take a long time to train-- is the tradeoff of time worth the performance? We utlized the Colorful Image Colorization model as a baseline to compare the results of our Pix2Pix model against. Instead of trying to determine the "true" colors that were present in grayscale image, the Colorful Image Colorization tries to generate a plausible coloring of the image and impose it onto the original instead of generating it from scratch. This model takes under 30 seconds to run, so it's a much more efficient alternative to Pix2Pix. We ran this model with a subset of the testing data from the NCD to see that while it performed well, pix2pix created a depiction that was closer to the true image. One of the baseline models, however, did perform better in terms of maintaining the same structural similarity to the original image. A sample image from NCD colored by different models is displated below.


<img src="https://user-images.githubusercontent.com/56567536/167536438-9a7b3524-6ead-42da-944e-e6ca25ebbb64.png">


### Experiment 5: Assessment of Performance on Landscapes Dataset
Since the Landscapes dataset was much larger than and incredibly different from NCD, we were curious to see how well a model that we had optimized for and trained on NCD would generate images given a grayscale landscape image. Due to time constraints as well as delays in debugging technical issues, we were unable to perform this experiment in time.


## Special FX Style Transfer
The goal of this experiment was to see if we could automatically remaster classic movies by training an unpaired GAN with screenshots from visually-similar movies from different eras. We chose the Star Wars series, as movies IV-VI and VII-IX depict similar things but were made over 30 years apart and thus feature very different visual effects. 

View best results [here](/SW_CycleGAN_fulldataset_512pix/index.html)!

### Data Pre-Processing
We used a script located in the `data` directory to chop the movies up into frames that we provided to the GAN. We also chose to downscale the images to speed up training time, since this is just a proof-of-concept.

### Training
Scripts to run all of these experiments are located in the `starwars_scripts directory`.

### Experiment 1: Running Open-Source Implementation 
For the first experiment, we simply ran the model on images downscaled to 286 pixels to produce some preliminary results.
See `starwars_scripts/test_SW_CycleGAN_reduceddataset_512pix.sh`

### Experiment 2: Higher Image Resolution
For our next experiment, we attempted to use higher-resolution images to see if that would make changes by the model more apparent. This experiment can be recreated by comparing the results of `starwars_scripts/train_SW_CycleGAN_reduceddataset_256pix.sh` and `starwars_scripts/train_SW_CycleGAN_fulldataset_512pix.sh`

### Experiment 3: Expanded Star Wars Dataset
Next, we expanded the size of the datset to include all 6 movies instead of just Episode IV and Episode VII. This led to much better results, shown on the results tab of this website. To recreate, run `starwars_scripts/train_SW_CycleGAN_reduceddataset_512pix.sh` and `starwars_scripts/train_SW_CycleGAN_fulldataset_512pix.sh`

### Experiment 4: Batch size and Longer Training
It turns out, increasing the batch size creates much blurrier images in the same amount of training time. If you want to see for yourself, `starwars_scripts/train_SW_CycleGAN_fulldataset_512_batch4pix.sh`

## References
Some useful references: 
- Anwar, S., Tahir, M., Li, C., Mian, A., Khan, F. S., & Muzaffar, A. W. (2020). Image colorization: A survey and dataset. arXiv preprint arXiv:2008.10774.

- Google. (2020, February 10). Common problems | generative adversarial networks | google developers. Google. Retrieved March 5, 2022, from [https://developers.google.com/machine-learning/gan/problems](https://developers.google.com/machine-learning/gan/problems)

- Greg, D. & Veerapaneni, R. (2018, January 10). Tricking Neural Networks: Create your own Adversarial Examples. Medium. Retrieved March 5, from [https://medium.com/@ml.at.berkeley/tricking-neural-networks-create-your-own-adversarial-examples-a61eb7620fd8](https://medium.com/@ml.at.berkeley/tricking-neural-networks-create-your-own-adversarial-examples-a61eb7620fd8)

- Lee, M. and Seok, J. (2020). Regularization Methods for Generative Adversarial Networks: An Overview of Recent Studies. ArXiv.

- Smith, T. (2019, October 23). Colorizing images with a convolutional neural network. Medium. Retrieved March 5, 2022, from [https://towardsdatascience.com/colorizing-images-with-a-convolutional-neural-network-3692d71956e2](https://towardsdatascience.com/colorizing-images-with-a-convolutional-neural-network-3692d71956e2)

- Žeger, I., Grgic, S., Vuković, J., & Šišul, G. (2021). Grayscale Image Colorization Methods: Overview and Evaluation. IEEE Access.

- Zhang, R., Isola, P., & Efros, A. A. (2016, October). Colorful image colorization. In European conference on computer vision (pp. 649-666). Springer, Cham.


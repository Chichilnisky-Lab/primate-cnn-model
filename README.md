# primate-cnn-model

## Overview
#### In this repository, we provide a convolutional neural network (CNN) light response model implementation applied to responses of primate retinal ganglion cells to visual stimulation with natural images. We also implement a LN model using the same stochastic gradient descent framework as used for the CNN model.

## Data preprocessing
#### Electrical recordings of primate RGCs during stimulation with natural images from the ImageNet database were acquired on a mutli-electrode array system at 20 kHz and spike sorting with KiloSort 2 was performed. Spikes were binned at 8.33 ms precision and smoothed with a Gaussian kernel (σ = 8.33 ms). The stimuli were 8 bit grayscaled images on a grid of size 160x320 and downsampled 2x to 80x160 for model training.

## File organization
#### ```./src/config.py```: a config file that contains hyperparameters to be set for the model as well as other constants used throughout the repository.
#### ```./src/Dataset.py```: dataloader class for the batch sampler used in mini-batch gradient descent.
#### ```./src/models.py```: model definition for the CNN/LN model.
#### ```./src/misc_util.py```: handful of functions that are used within the dataset class and model definition.
#### ```./scripts/train_model.py```: contains a training script that loads the preprocessed data, performs mini-batch gradient descent and writes model files to disk after each training epoch.

#### This repository is accompanied by a preprint on bioRxiv: https://www.biorxiv.org/content/10.1101/2024.03.22.586353.abstract. Please see the manuscript for additional details.

#### Authors: Alex Gogliettino and Sam Cooler (Chichilnisky Lab, 2024). We would also like to acknowledge Joshua Melander, Steve Baccus and other members from the Baccus Lab, who provided inspiration and guidance on the model with their Deep Retina project.

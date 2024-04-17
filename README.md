# Denoising Experiment with PMCnet Algorithm

In this repository, I use the PMCnet algorithm proposed by Huang et al. to perform denoising experiments using the MNIST dataset.

## Notebooks
- **PMCnet_denoiser.ipynb**: This notebook contains the implementation of the PMCnet algorithm for training a denoiser using the MNIST dataset.
- **denoising_autoencoder.ipynb**: This notebook contains the implementation of Maximum Likelihood Estimation (MLE) training for denoising using an autoencoder approach.
- **multiple_denois**: This notebook contains the implementatino of the PMCnet algorithm for training a denoiser on different noise levels.
- **denoising_diffusion_model**: This notebook contains two trivial denoising-diffusion like implementations using PMCnet trained denoising blocks.
- **PMCnet_regression**: This notebook contains the regression experiment from Huang et al. on the Naval dataset.

## Scripts
- **DenoisingNetwork.py**: This script contains the Denoising Autoencoder.
- **PMCnet.py**: This script contains the PMCnet algorithm.
- **functions.py**: This script contains functions used throughout the scripts and notebooks.
  
Feel free to explore the notebooks and experiment with the code!

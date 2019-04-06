# adversarial-medicine

This repository contains the code used to train the proof-of-concept models and adversarial attacks described in ["Adversarial Attacks Against Medical Deep Learning Systems"](https://arxiv.org/abs/1804.05296), an updated version of which is under consideration at a CS venue.

## Data 

### Raw Images
All of the data used in the project is publically available, as outlined in the paper.  The original sources are below:

[CXR Data](http://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a)

[DR Data](https://www.kaggle.com/c/diabetic-retinopathy-detection)

[Melanoma Data](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery)

As outlined in the paper, regardless of any train/test splits in the original datasets, I merged all the images together and split by patient into ~80/20 train/test splits.  The DR Kaggle repo in particular was *way* too test-heavy in their train/test split for my purposes.


### Numpy Arrays

To make it easier to recreating the results in this repo, I also provide numpy arrays for the validation sets for each of the above datasets [here](https://www.dropbox.com/sh/tg6xij9hhfzgio9/AADqu6BMq3Rko7U7-q6vwmMFa?dl=0).  (Training sets are too big).  These numpy validation sets, along with the keras models below, are sufficient to run the Jupyter notebooks generating the figures and python script generating the tables.

## Models

### Pretrained Keras models

If you want to skip the training steps, I provide the keras models (in the case of white-box models) and keras model weights (for the separately trained black-box models) for each of the three tasks [here](https://www.dropbox.com/sh/8a9j9773c1sejol/AAAEvXDafJCPbq5YOBRG4wx0a?dl=0).

Note: *please* don't try to use these for any medical purposes. I noticed some big problems in these datasets and do not trust them for more than proofs of concept.

### Code to recreate the models

[train_model.py](train_models/train_model.py) is a stand-alone python script that will train and save a model for one of the tasks.  The model assumes that images are organized in folders at the locations `images/train`and `images/val` relative to the python script's working directory.  Alternatively, numpy data blobs of all the images in your training/test sets can be placed at (`data/train_x.npy`, 'data/train_y.npy', 'data/test_x.npy', 'data/test_y.npy').

The training script has a number of options, including whether to train inception or resnet models (in the paper I only report results for Resnet for simplicity, but results on Inception were ~identical), learning rates, early stopping, various data augmentations, etc.  In practice, I got good enough results with the defaults I loaded in here so I didn't do much by way of hyperparameter tuning, so I'm sure a lot more performance could get squeezed out of the models.

Dependencies: requires Python3, keras, tensorflow, and numpy.

## Recreating PGD Attacks and Figures

The `pgd_attacks` directory contains a [craft_attacks.py](pgd_attacks/craft_attacks.py) script that builds the attacks, reports their success metrics, and saves the adversarial attacks themselves.  This file assumes the white box model is located in `models/wb_model.h5` and the weights for the black box model are located in `models/bb_weights.hdf5"`, though there are command-line arguments to specify different locations.

The file [generate_pgd_figures.ipynb](pgd_attacks/generate_pgd_figures.ipynb) is a Jupyter notebook that recreates the figures in the paper based on the files produced by `craft_attacks.py`.

Dependencies: see [pgd_environment.yaml](pgd_attacks/pgd_environment.yaml)


## Recreating Patch Attacks and Figures 

The patch attacks were largely developed by [Hyung Won Chung](https://github.com/hwc27). The bulk of the functionality is developed in [execute_patch_attacks.py](patch_attacks/execute_patch_attacks.py) and [craft_attack_patch.py](patch_attacks/craft_attack_patch.py).  

There are two options to explore the patch attack: Jupyter Notebook interface and directly running [craft_attack_patch.py](patch_attacks/craft_attack_patch.py).

1) Notebook
The file [0_generate_patch_results_derm.ipynb](patch_attacks/0_generate_patch_results_derm.ipynb) is a Jupyter notebook that generates the figures for the derm patches.  The similarly named [1_generate_patch_results_cxr.ipynb](patch_attacks/1_generate_patch_results_cxr.ipynb) and [1_generate_patch_results_dr.ipynb](patch_attacks/1_generate_patch_results_dr.ipynb) are similar notebooks that generate the results for DR and Chest-xrays. 

2) [craft_attack_patch.py](patch_attacks/craft_attack_patch.py).
This file can be used stand-alone and does not require any command-line argument. So just run it via `python craft_attack.py`. To faciliate testing the functionality, we have included sample train and test images (8 images for each label) for Melanoma. For training the patch, use the full [Data](#data). We couldn't include the pretrained models object due to size. Create a directory `models` and put the model objects. For example, download the [Pretrained Keras models](#pretrained-keras-models) and put both files in the `models` directory.

Dependencies: requires Python3, keras, tensorflow, cleverhans, numpy, scipy, sklearn.

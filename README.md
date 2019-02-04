# adversarial-medicine

This repository contains the code used to train the proof-of-concept models and adversarial attacks described in ["Adversarial Attacks Against Medical Deep Learning Systems"](https://arxiv.org/abs/1804.05296), an updated version of which is under consideration at a CS venue.

## Data 

### Raw Images
All of the data used in the project is publically available, as outlined in the paper.  However, to spare people the pain, I'll provide links to the data folders

The data is far too large for Github, so I am provided it in the form of dropbox links.

[CXR Data](https://www.dropbox.com/sh/w0r19j8fd2d7m33/AAAFPDzux_aWQsFP_a0f498Ta?dl=0)

[DR Data](https://www.dropbox.com/sh/tpeh0ktsurzubz5/AAAVPMKA-FaRDlGeyRfGn-vXa?dl=0)

[Melanoma Data](https://www.dropbox.com/sh/bryrme8sr0ry091/AABtSGGzjBf5UIr8Ae-G6gdva?dl=0)


Each of the above folders follows the same subdirectory structure:

`train/0_control`

`train/1_case`

`val/0_control`

`val/1_case`

where the "0_control" subdirectories contain the individual image files with pneumothorax, referable retinopathy, or melanoma.

Important note:  Please do not use this data to build clinically deployable models.  I was using the data as a proof of concept, and as such didn't fuss too much about verifying the accuracy of the labels, but couldn't help but stumble upon a number of errors in the provided labels.

### Numpy Arrays

I provide numpy arrays for the validation sets for each of the above datasets [here](https://www.dropbox.com/sh/tg6xij9hhfzgio9/AADqu6BMq3Rko7U7-q6vwmMFa?dl=0).  (Training sets are too big).


## Training the models

[train_model.py](train_models/train_model.py) is a stand-alone python script that will train and save a model for one of the tasks.  To run the model, simply place any of the above data folders at the location `images/train`and `images/val` relative to the python script's working directory.  Alternatively, numpy data blobs of all the images in your training/test sets can be placed at (`data/train_x.npy`, 'data/train_y.npy', 'data/test_x.npy', 'data/test_y.npy').

The training script has a number of options, including whether to train inception or resnet models (in the paper I only report results for Resnet for simplicity, but results on Inception were identical), learning rates, early stopping, various data augmentations, etc.  In practice, I got good enough results with the defaults I loaded in here so I didn't do much by way of hyperparameter tuning, but I include more functionality in case it's helpful.

In addition, I provide the models (in the case of white-box models) and model weights (for the separately trained black-box models) for each of the three models [here](https://www.dropbox.com/sh/8a9j9773c1sejol/AAAEvXDafJCPbq5YOBRG4wx0a?dl=0).

Dependencies: requires Python3, keras, tensorflow, and numpy.


## Recreating PGD Attacks and Figures

The `pgd_attacks` directory contains a [craft_attacks.py](pgd_attacks/craft_attacks.py) script that builds the attacks, reports their success metrics, and saves the adversarial attacks themselves.  This file assumes the white box model is located in `models/wb_model.h5` and the weights for the black box model are located in `models/bb_weights.hdf5"`, though there are command-line arguments to specify different locations.

The file [generate_pgd_figures.ipynb](pgd_attacks/generate_pgd_figures.ipynb) is a Jupyter notebook that recreates the figures in the paper based on the files produced by `craft_attacks.py`.

Dependencies: requires Python3, keras, tensorflow, cleverhans, numpy, scipy, sklearn.


## Recreating Patch Attacks and Figures 

The patch attacks were largely developed by my colleague [Hyung Won Chung](https://scholar.google.com/citations?user=1CAlXvYAAAAJ&hl=en).  The bulk of the functionality is developed in [execute_patch_attacks.py](patch_attacks/execute_patch_attacks.py) and [patch_derm_v3.py](patch_attacks/patch_derm_v3.py).  The file [0_generate_patch_results_derm.ipynb](patch_attacks/0_generate_patch_results_derm.ipynb) is a Jupyter notebook that generates the figures for the derm patches.  The similarly named [1_generate_patch_results_cxr.ipynb](patch_attacks/1_generate_patch_results_cxr.ipynb) and [1_generate_patch_results_dr.ipynb](patch_attacks/1_generate_patch_results_dr.ipynb) are similar notebooks that generate the results for DR and Chest-xrays. 

Dependencies: requires Python3, keras, tensorflow, cleverhans, numpy, scipy, sklearn.

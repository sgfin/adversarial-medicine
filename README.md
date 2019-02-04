# adversarial-medicine

This repository contains the code used to train the proof-of-concept models and adversarial attacks described in ["Adversarial Attacks Against Medical Deep Learning Systems"](https://arxiv.org/abs/1804.05296), an updated version of which is under consideration at a CS venue.

## Data Access

All of the data used in the project is publically available, as outlined in the paper.  However, to spare people the pain, I'll provide links to the data folders

The data is far too large for Github, so I am provided it in the form of dropbox links.

[CXR Data](https://www.dropbox.com/sh/w0r19j8fd2d7m33/AAAFPDzux_aWQsFP_a0f498Ta?dl=0)
[DR Data](https://www.dropbox.com/sh/tpeh0ktsurzubz5/AAAVPMKA-FaRDlGeyRfGn-vXa?dl=0)
[Melanoma Data](https://www.dropbox.com/sh/bryrme8sr0ry091/AABtSGGzjBf5UIr8Ae-G6gdva?dl=0)

Each of the above folders is organized in the same way:

`train/0_control`
`train/1_case`
`val/0_control`
`val/1_case`

where the "0_control" subdirectories contain the image files with pneumothorax, referable retinopathy, or melanoma.


## 

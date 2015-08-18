# Introduction

Project as part of Dropbox's Hack Week to provide a classifier for detecting clouds in remote sensing data using deep learning.

# Details

Datasets are in data/landsat/images and data/planetlab/images; however, these are not checked in due to possible licensing issues. Metadata that we've added via annotation in order to label the data _is_ checked in and is in data/landsat/metadata and data/planetlab/metadata.

Preparing the data, training, and generating graphs to know how we are doing is via a single command-line tool. Run the following to see options:

./src/cloudless/cloudless.py --help

Note that you must have CAFFE_HOME defined before running this with Caffe installed.

Training info and graphs go into logs/.

We currently have pretrained weights from the BVLC AlexNet Caffe Model Zoo, in src/caffe_model/bvlc_alexnet. This is trained on ILSVRC 2012, almost exactly as described in [ImageNet classification with deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) by Krizhevsky et al. in NIPS 2012.

Note that the trained AlexNet file is much too large to check into Github (it's about 350MB). You will have to download the file from [here](http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel) and copy it to src/caffe_model/bvlc_alexnet/bvlc_alexnet.caffemodel.

This is all a stub for now; nothing real will happen currently.

# Landsat data details

The scripts to prepare the Landsat data are [here](https://github.com/max-nova/cloudless).

The zip file is the raw imagery stuff pulled down from USGS.

The training-set.csv has 5 columns:
 * Image Name
 * Clouds - 1 if there are clouds
 * Edge - 1 if the image is partially nulled out
 * Blank - 1 if the image is totally nulled out
 * Comments - any random comments I had

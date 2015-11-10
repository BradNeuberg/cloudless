# Introduction

This project provides a classifier for detecting clouds in satellite remote sensing data using deep learning. Startups like [Planet Labs](https://www.planet.com/) are launching fleets of nanosats to image much of the earth daily; detecting clouds in these images to ignore or eliminate them is an important pre-processing step to doing interesting work nanosat imagery.

This project has three parts:

* An annotation tool that takes data from the [Planet Labs API](https://www.planet.com/docs/) and allows users to draw bounding boxes around clouds.
* A training pipeline that takes annotated data, runs it on EC2 on GPU boxes to fine tune an AlexNet trained model, and then generates validation statistics to relate how well the trained model performs.
* A bounding box system that takes the trained cloud classifier and attempts to draw bounding boxes on orbital satellite data.

This project and its trained model are available under an Apache 2 license; see the license.txt file for details.

Parts of the Cloudless project started as part of Dropbox's Hack Week, with continued work post-Hack Week by Brad Neuberg. Contributors:
* Johann Hauswald
* Max Nova
* Brad Neuberg

# Data

Preprocessed datasets are in data/planetlab/images while metadata added via annotation is in data/planetlab/metadata; however, these are not checked in due to possible licensing issues. Data that has been processed into training and validation datasets are saved as LevelDB files into data/leveldb; these are also not checked in due to size and licensing issues.

# Annotation Tool

This currently has its own README file at [src/annotate/README.md](src/annotate/README.md).

# Training Pipeline

Preparing the data, training, and generating graphs to know how we are doing is via a single command-line tool, cloudless.py, located in src/cloudless/train.

To setup this tool, you must have CAFFE_HOME defined and have Caffe installed with the Caffe Python bindings setup.

Second, ensure you have all Python requirements installed by going into the cloudless root directory and running:

    pip install -r requirements.txt

Third, ensure you have ./src in your PYTHONPATH as well as the Python bindings for Caffe compiled and in your PYTHONPATH as well:
    export PYTHONPATH=$PYTHONPATH:/usr/local/caffe/python:./src

Run the following to see options:

    ./src/cloudless/cloudless.py --help

Training info and graphs go into logs/.

We currently have pretrained weights from the BVLC AlexNet Caffe Model Zoo, in src/caffe_model/bvlc_alexnet. This is trained on ILSVRC 2012, almost exactly as described in [ImageNet classification with deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) by Krizhevsky et al. in NIPS 2012.

Note that the trained AlexNet file is much too large to check into Github (it's about 350MB). You will have to download the file from [here](http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel) and copy it to src/caffe_model/bvlc_alexnet/bvlc_alexnet.caffemodel.

A trained, fine tuned model is available on S3 [here](https://s3.amazonaws.com/cloudless-data/bvlc_alexnet_finetuned.caffemodel). Download this and place it into src/caffe_model/bvlc_alexnet/bvlc_alexnet_finetuned.caffemodel. It's current accuracy is 62.50%, while its F1 score is 0.65. See [logs/output0003.statistics.txt](logs/output003.statistics.txt) for full accuracy details.

Once you've prepped your datasets, the first step is to preprocess the data into the format required by Caffe, LevelDB:

    ./src/cloudless/cloudless.py -p

Then you can train and generate validation statistics and logs:

    ./src/cloudless/cloudless.py -t -g

This will output various graphs into the logs/ directory, in increasing numbers (i.e. output0003.log, output0004.log, etc.).

# Bounding Box System

This currently has its own README file in [src/cloudless/inference/README.md](src/cloudless/inference/README.md).

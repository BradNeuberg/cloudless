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

There are a series of Python programs to aid in preparing data for training, doing the actual training, and then seeing how well the trained model performs via graphs and statistics, all located in src/cloudless/train.

To setup these tools, you must have CAFFE_HOME defined and have Caffe installed with the Caffe Python bindings setup.

Second, ensure you have all Python requirements installed by going into the cloudless root directory and running:

    pip install -r requirements.txt

Third, ensure you have ./src in your PYTHONPATH as well as the Python bindings for Caffe compiled and in your PYTHONPATH as well:

    export PYTHONPATH=$PYTHONPATH:/usr/local/caffe/python:./src

Note: for any of the data preparation, training, or graphing Python scripts below you can add `--help` to see what command-line options are available to override defaults.

To prepare data that has been labelled via the [annotation tool](src/annotate/README.md), first run the following from the root directory:

    ./src/cloudless/train/prepare_data.py --input_metadata data/planetlab/metadata/annotated.json --input_images data/planetlab/metadata --output_images data/planetlab/metadata/bounded --output_leveldb data/leveldb

TODO: Have a command line option to do data augmentation.

To train using the prepared data, run the following from the root directory:

    ./src/cloudless/train/train.py --log_num 1

You can keep incrementing the `--log_num` option while doing test runs in order to have log output get saved for each session for later analysis. By default this will place the trained, fine-tuned model into `logs/latest_bvlc_alexnet_finetuned.caffemodel`; this can be changed via the `--output_weight_file` option.

To generate graphs and verify how well the trained model is performing (note that you should set the log number to be the same as what you set it for `train.py`):

    ./src/cloudless/train/test.py --log_num 1 --note "This will get added to graph"

Use the `--note` property to add extra info to the quality graphs, such as various details on hyperparameter settings, so you can reference them in the future.

You can also predict how well the trained classifier is doing on a single image via the `predict.py` script:

    ./src/cloudless/train/predict.py --image examples/cloud.png
    ./src/cloudless/train/predict.py --image examples/no_cloud.png

The four scripts above all have further options to customize them; add `--help` as an option when running them.


Training info and graphs go into logs/.

We currently have pretrained weights from the BVLC AlexNet Caffe Model Zoo, in src/caffe_model/bvlc_alexnet. This is trained on ILSVRC 2012, almost exactly as described in [ImageNet classification with deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) by Krizhevsky et al. in NIPS 2012.

Note that the trained AlexNet file is much too large to check into Github (it's about 350MB). You will have to download the file from [here](http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel) and copy it to src/caffe_model/bvlc_alexnet/bvlc_alexnet.caffemodel.


A trained, fine tuned model is available on S3 [here](https://s3.amazonaws.com/cloudless-data/bvlc_alexnet_finetuned.caffemodel). Download this and place it into src/caffe_model/bvlc_alexnet/bvlc_alexnet_finetuned.caffemodel. It's current accuracy is 62.50%, while its F1 score is 0.65. See [logs/output0003.statistics.txt](logs/output003.statistics.txt) for full accuracy details.

# Bounding Box System

This currently has its own README file in [src/cloudless/inference/README.md](src/cloudless/inference/README.md).

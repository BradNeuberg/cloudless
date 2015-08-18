# Introduction

Project as part of Dropbox's Hack Week to provide a classifier for detecting clouds in remote sensing data using deep learning.

# Details

Datasets are in data/landsat/images and data/planetlab/images; however, these are not checked in due to possible licensing issues. Metadata that we've added via annotation in order to label the data _is_ checked in and is in data/landsat/metadata and data/planetlab/metadata.

Preparing the data, training, and generating graphs to know how we are doing is via a single command-line tool. Run the following to see options:

./src/cloudless/cloudless.py --help

Note that you must have CAFFE_HOME defined before running this with Caffe installed.
This is all a stub for now; nothing real will happen currently.

# Introduction

This project provides a classifier for detecting clouds in satellite remote sensing data using deep learning. Startups like [Planet Labs](https://www.planet.com/) are launching fleets of nanosats to image much of the earth daily; detecting clouds in these images to ignore or eliminate them is an important pre-processing step to doing interesting work nanosat imagery.

This project has three parts:

* An annotation tool that takes data from the [Planet Labs API](https://www.planet.com/docs/) and allows users to draw bounding boxes around clouds.
* A training pipeline that takes annotated data, runs it on EC2 on GPU boxes to fine tune an AlexNet trained model, and then generates validation statistics to relate how well the trained model performs.
* A bounding box system that takes the trained cloud classifier and attempts to draw bounding boxes on orbital satellite data.

This project and its trained model are available under an Apache 2 license; see the license.txt file for details.

Parts of the Cloudless project started as part of Dropbox's Hack Week, with continued work post-Hack Week by Brad Neuberg. Contributors:
* Brad Neuberg
* Johann Hauswald
* Max Nova

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

    ./src/cloudless/train/prepare_data.py --input_metadata data/planetlab/metadata/annotated.json --input_images data/planetlab/metadata --output_images data/planetlab/metadata/bounded --output_leveldb data/leveldb --log_num 1

You can keep incrementing the `--log_num` option while doing data preparation and test runs in order to have log output get saved for each session for later analysis.

TODO: Have a command line option to do data augmentation.

To train using the prepared data, run the following from the root directory:

    ./src/cloudless/train/train.py --log_num 1

By default this will place the trained, fine-tuned model into `logs/latest_bvlc_alexnet_finetuned.caffemodel`; this can be changed via the `--output_weight_file` option.

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

# Training on AWS

The code base includes scripts for training on Amazon Web Services (AWS) machines that have GPUs.

To use, first make sure you provision an S3 bucket named `cloudless-data`. Training results will be dumped into there.

Next, you will need to get an AMI (Amazon Machine Image) that is configured with Caffe and CUDA. You can get a publicly available one from here based on Ubuntu:

https://github.com/BVLC/caffe/wiki/Caffe-on-EC2-Ubuntu-14.04-Cuda-7

Next, launce an instance using your AMI; I suggest an g2.2xlarge instance that has a GPU but which is cheaper than a full g2.8xlarge which isn't really needed for cloudless currently. When you launch the instance make sure to also add EBS storage so that your data can persist between runs. This is useful so you don't have to re-upload all your data and configure cloudless for each training run of the model.

Make sure you can SSH into your new instance; it's beyond the scope of this document to describe how to do this. Also make sure to setup the [EC2 CLI tools](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-set-up.html).

Set `EC2_KEYPAIR` to where the PEM keypair is for your EC2 instance and `EC2_HOST_NAME` to the public host name of your launched instance:

    export EC2_KEYPAIR=/Users/bradneuberg/Dropbox/AWS/ami-generated-keypair-2.pem
    export EC2_HOST_NAME=ec2-52-90-165-78.compute-1.amazonaws.com

The first time you setup your instance you will need to copy cloudless and your training data over onto your EBS volume. Before you do, make sure you've already prepared your data into leveldb databases as detailed earlier in this document for the `prepare_data.py` script _outside_ of your VM.

Next, _outside_ your AWS instance go into your `cloudless/` directory checkout of the git source and run the following to copy over the cloudless source and all your training data in `data/planetlab` (assuming you have generated annotated training data, which is not bundled with the git repo due to the raw data belonging to Planet Lab):

    ./src/aws_scripts/aws_rsync.sh $EC2_KEYPAIR $EC2_HOST_NAME

Now SSH into your AWS instance and configure it so that we can shutdown the instance without using sudo, which will be needed later for the `--terminate` option to work on the `train.sh` script below:

    ssh -i $EC2_KEYPAIR ubuntu@$EC2_HOST_NAME
    sudo chmod a+s /sbin/shutdown

You can now train the model on your AWS instance, using the `screen` command to ensure training will last even if you quit SSH:

    screen -S training_session_1
    cd /data/cloudless
    ./src/aws_scripts/train.sh --log_num 1 --note "Testing training run" --terminate
    # Press control-a d to leave screen session running

Change the `--log_num` value to the number you want appended to log output. `--note` is required and will be printed into the log file; it's an appropriate place to put details such as hyperparameters being experimented with for this training run. `--terminate` is optional, and if present will shut down the AWS instance when training is finished in order to save money.

You can re-connect to a screen session to see how training is going:
    ssh -i $EC2_KEYPAIR ubuntu@$EC2_HOST_NAME
    screen -x training_session_1

Note: The `train.sh` script is currently hard-coded to use S3 instances in the `us-east-1` region; change `S3_REGION` inside the script if your setup differs.

When training is finished the results will end up in the `cloudless-data` bucket on S3, tarred and gzipped. You can download this and run it locally on your host against the test validation scripts to see how well training went. On your own machine _outside_ aws run:

    aws s3 ls cloudless-data

The result should be the latest GZIP file; if you are running several tests in parallel make sure to look at the private instance IP address in the GZIP file name to match it up with your particular run.

Run the following to generate validation graphs (with example GZIP filename), from your `cloudless` checked out directory _outside_ the AWS instance:
    aws s3 cp s3://cloudless-data/caffe-results-12-16-15-host-ip-172-31-6-33-time-1450242072.tar.gz ~/tmp
    gunzip ~/tmp/caffe-results-12-16-15-host-ip-172-31-6-33-time-1450242072.tar.gz
    tar -xvf ~/tmp/caffe-results-12-16-15-host-ip-172-31-6-33-time-1450242072.tar
    ./src/cloudless/train/test.py --log_num 1 --note "Testing training run"

# Bounding Box System

This currently has its own README file in [src/cloudless/inference/README.md](src/cloudless/inference/README.md).

# Introduction

This project provides a classifier for detecting clouds in satellite remote sensing data using deep learning. Startups like [Planet Labs](https://www.planet.com/) have launched fleets of nanosats to image much of the earth daily; detecting clouds in these images to ignore or eliminate them is an important pre-processing step to doing interesting work nanosat imagery.

This project has three parts:

* An annotation tool that takes data from the [Planet Labs API](https://www.planet.com/docs/) and allows users to draw bounding boxes around clouds to bootstrap training data.
* A training pipeline that takes annotated data, runs it on EC2 on GPU boxes to fine tune an AlexNet trained model, and then generates validation statistics to relate how well the trained model performs.
* A bounding box system that takes the trained cloud classifier and attempts to draw bounding boxes on orbital satellite data.

Example output of before and after images with detected clouds with yellow overlay boxes via the trained neural network shown below:

![normal image for comparison](examples/rapideye_cloud_2.jpg "Normal cloud image for comparison")

![cloud detection boxes](examples/rapideye_cloud_2-regions.png "Areas with yellow boxes are clouds")

Note that even though Cloudless is currently focused on cloud detection and localization, the entire pipeline can be used for any other satellite detection task with just a bit of tweaking, such as detecting cars, different biomes, etc. Use the annotation tools to bootstrap training data then run it through the pipeline for your particular task; everything in Cloudless is what you would need for other kinds of orbital computer vision detection tasks.

This project and its trained model are available under an Apache 2 license; see the [license.txt file](license.txt) for details.

Parts of the Cloudless project started as part of Dropbox's Hack Week, with continued work post-Hack Week by Brad Neuberg. Contributors:
* [Brad Neuberg](http://codinginparadise.org)
* [Johann Hauswald](http://web.eecs.umich.edu/~jahausw/)
* [Max Nova](https://www.linkedin.com/in/maxnova)

This is release 1.0 of Cloudless.

# Annotation Tool

The annotation tool makes it possible to label Planet Labs data to feed into the neural network. It's code lives in [src/annotate](src/annotate).

Setting it up:
* brew install gdal
* Install virtualenv and virtualenvwrapper: https://jamie.curle.io/posts/installing-pip-virtualenv-and-virtualenvwrapper-on-os-x/
* mkvirtualenv annotate-django
* cd src/annotate
* pip install -r requirements.txt
* ./manage.py migrate
* echo 'PLANET_KEY="SECRET PLANET LABS KEY"' >> $VIRTUAL_ENV/bin/postactivate

Each time you work with the annotation tool you will ned to re-activate its virtualenv setup:

```
workon annotate-django
```

When finished working run:

```
deactivate
```

To import imagery into the annotation tool, go into `src/annotate` and:

1. Choose your lat/lng and buffer distance (meters) you want (this example is for San Fran) and the directory to download to, then run:

```
python train/scripts/download_planetlabs.py 37.796105 -122.461349 --buffer 200 --image_type rapideye --dir ../../data/planetlab/images/
```

2. Chop up these raw images into 512x512 pixels and add them to the database

```
./manage.py runscript populate_db --script-args ../../data/planetlab/images/ 512
```

To begin annotating imagery:

1. Start the server running:

```
./manage.py runserver
```

2. Go to http://127.0.0.1:8000/train/annotate

3. Draw bounding boxes on the image.

4. Hit the "Done" button to submit results to the server

5. Upon successful submission, the browser will load a new image to annotate

To export annotated imagery so it can be consumed by the training pipeline:

1. Writes out annotated.json and all the annotated images to a specified directory

```
./manage.py runscript export --script-args ../../data/planetlab/metadata/
```

If you need to clear out the database and all its images to restart for some reason:

```
./manage.py runscript clear
```

# Training Pipeline

Once you've annotated images using the annotation tool, you can bring them into the training pipeline to actually train a neural network using Caffe and your data. Note that the training pipeline does not use virtualenv, so make sure to de-activate any virtualenv environment you've activated for the annotation tool earlier.

There are a series of Python programs to aid in preparing data for training, doing the actual training, and then seeing how well the trained model performs via graphs and statistics, all located in [src/cloudless/train](src/cloudless/train).

To setup these tools, you must have CAFFE_HOME defined and have Caffe installed with the Caffe Python bindings setup.

Second, ensure you have all Python requirements installed by going into the cloudless root directory and running:

```
pip install -r requirements.txt
```

Third, ensure you have ./src in your PYTHONPATH as well as the Python bindings for Caffe compiled and in your PYTHONPATH as well:

```
export PYTHONPATH=$PYTHONPATH:/usr/local/caffe/python:./src
```

Original raw TIFF imagery that will be fed into the annotation tool should go into data/planetlab/images while metadata added via annotation is in data/planetlab/metadata. Data that has been prepared by one of the data prep tools below are saved as LevelDB files into data/leveldb. Raw data is not provided due to copyright concerns; however, access to some processed annotation data is available. See the section "Trained Models and Archived Data" below for more details.

Note: for any of the data preparation, training, or graphing Python scripts below you can add `--help` to see what command-line options are available to override defaults.

To prepare data that has been labelled via the annotation tool, first run the following from the root directory:

```
./src/cloudless/train/prepare_data.py --input_metadata data/planetlab/metadata/annotated.json --input_images data/planetlab/metadata --output_images data/planetlab/metadata/bounded --output_leveldb data/leveldb --log_num 1
```

You can keep incrementing the `--log_num` option while doing data preparation and test runs in order to have log output get saved for each session for later analysis. If `--do_augmentation` is it present we augment the data with extra training data manual 90 degree rotations. Testing found, however, that these degrade performance rather than aid performance.

To train using the prepared data, run the following from the root directory:

```
./src/cloudless/train/train.py --log_num 1
```

By default this will place the trained, fine-tuned model into `logs/latest_bvlc_alexnet_finetuned.caffemodel`; this can be changed via the `--output_weight_file` option.

To generate graphs and verify how well the trained model is performing (note that you should set the log number to be the same as what you set it for `train.py`):

```
./src/cloudless/train/test.py --log_num 1 --note "This will get added to graph"
```

Use the `--note` property to add extra info to the quality graphs, such as various details on hyperparameter settings, so you can reference them in the future.

You can also predict how well the trained classifier is doing on a single image via the `predict.py` script:

```
./src/cloudless/train/predict.py --image examples/cloud.png
./src/cloudless/train/predict.py --image examples/no_cloud.png
```

The four scripts above all have further options to customize them; add `--help` as an option when running them.

Training info and graphs go into logs/.

### Trained Models and Archived Data

We currently have pretrained weights from the BVLC AlexNet Caffe Model Zoo, in src/caffe_model/bvlc_alexnet. This is trained on ILSVRC 2012, almost exactly as described in [ImageNet classification with deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) by Krizhevsky et al. in NIPS 2012.

Note that the trained AlexNet file is much too large to check into Github (it's about 350MB). You will have to download the file from [here](http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel) and copy it to src/caffe_model/bvlc_alexnet/bvlc_alexnet.caffemodel.

A trained, fine tuned model is available in a shared Dropbox file folder [here](https://www.dropbox.com/sh/dbnc6y7abv1mt3i/AADhtAUQ6P7tgLfnEZGbdwsca?dl=0). Download this and place it into src/caffe_model/bvlc_alexnet/bvlc_alexnet_finetuned.caffemodel. It's current accuracy is 89.69%, while its F1 score is 0.91. See [logs/output0005.statistics.txt](logs/output005.statistics.txt) for full accuracy details. Note that the trained model is available under the same Apache 2 license as the rest of the code.

The [same shared Dropbox folder](https://www.dropbox.com/sh/dbnc6y7abv1mt3i/AADhtAUQ6P7tgLfnEZGbdwsca?dl=0) also has the training and validation leveldb databases. Trained models for earlier training runs can be found [here](https://www.dropbox.com/sh/xehr6ug9vhf2d97/AAB2Gt3lgCmbXW1w8EYlwQkpa?dl=0), while labelled RapidEye annotation data can be found [here](https://www.dropbox.com/sh/boosyg7ccnlij57/AAAXrkVfA1XQCUvjIqdULaCha?dl=0). Note that the annotated RapidEye imagery is downgraded and chopped up for serving by the annotation tool; nevertheless the imagery is owned by Planet Labs and is provided for reference only. The original raw RapidEye imagery fed into the annotation tool is not publicly available due to being owned by Planet Labs and is in a Dropbox folder (cloudless_data/original_tiffs for future reference).

A lab notebook with notes during training runs is at [logs/cloudless_lab_notebook.txt](logs/cloudless_lab_notebook.txt).

### Training on AWS

The code base includes scripts for training on Amazon Web Services (AWS) machines that have GPUs.

To use, first make sure you provision an S3 bucket named `cloudless-data`. Training results will be dumped into there.

Next, you will need to get an AMI (Amazon Machine Image) that is configured with Caffe and CUDA. You can get a publicly available one from here based on Ubuntu:

https://github.com/BVLC/caffe/wiki/Caffe-on-EC2-Ubuntu-14.04-Cuda-7

Next, launce an instance using your AMI; I suggest an g2.2xlarge instance that has a GPU but which is cheaper than a full g2.8xlarge which isn't really needed for cloudless currently. When you launch the instance make sure to also add EBS storage so that your data can persist between runs. This is useful so you don't have to re-upload all your data and configure cloudless for each training run of the model.

Make sure you can SSH into your new instance; it's beyond the scope of this document to describe how to do this. Also make sure to setup the [EC2 CLI tools](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-set-up.html).

Set `EC2_KEYPAIR` to where the PEM keypair is for your EC2 instance and `EC2_HOST_NAME` to the public host name of your launched instance:

```
export EC2_KEYPAIR=/Users/bradneuberg/Dropbox/AWS/ami-generated-keypair-2.pem
export EC2_HOST_NAME=ec2-52-90-165-78.compute-1.amazonaws.com
```

The first time you setup your instance you will need to copy cloudless and your training data over onto your EBS volume. Before you do, make sure you've already prepared your data into leveldb databases as detailed earlier in this document for the `prepare_data.py` script _outside_ of your VM.

Next, _outside_ your AWS instance go into your `cloudless/` directory checkout of the git source and run the following to copy over the cloudless source and all your training data in `data/planetlab` (assuming you have generated annotated training data, which is not bundled with the git repo due to the raw data belonging to Planet Lab).

You will also need to format and mount your EBS volume; in the code below change `/dev/xvdb` to the non-root EBS volume you setup:

```
sudo mkfs -t ext4 /dev/xvdb
sudo mkdir /data
sudo mount /dev/xvdb /data
sudo chown -R ubuntu /data
mkdir -p /data/cloudless/data/planetlab/metadata
mkdir -p /data/cloudless/data/leveldb
mkdir -p /data/cloudless/logs
mkdir -p /data/snapshots
```

You will also want to ensure this volume gets mounted when the machine restarts:

```
sudo vim /etc/fstab
```

Add a line like the following:

```
/dev/xvdb       /data   auto    defaults,nobootwait,nofail,comment=cloudconfig  0       2
```

Now SSH into your AWS instance and configure it so that we can shutdown the instance without using sudo, which will be needed later for the `--terminate` option to work on the `train.sh` script below:

```
ssh -i $EC2_KEYPAIR ubuntu@$EC2_HOST_NAME
sudo chmod a+s /sbin/shutdown
```

Now outside the EC2 instance on your host go into the cloudless directory and copy everything over to the EC2 instance:

```
./src/aws_scripts/aws_rsync.sh $EC2_KEYPAIR $EC2_HOST_NAME
```

You can now train the model on your AWS instance, using the `screen` command to ensure training will last even if you quit SSH:

```
screen -S training_session_1
cd /data/cloudless
./src/aws_scripts/train.sh --log_num 1 --note "Testing training run" --terminate
# Press control-a d to leave screen session running
```

Change the `--log_num` value to the number you want appended to log output. `--note` is required and will be printed into the log file; it's an appropriate place to put details such as hyperparameters being experimented with for this training run. `--terminate` is optional, and if present will shut down the AWS instance when training is finished in order to save money.

You can re-connect to a screen session to see how training is going:

```
ssh -i $EC2_KEYPAIR ubuntu@$EC2_HOST_NAME
screen -x training_session_1
```

Note: The `train.sh` script is currently hard-coded to use S3 instances in the `us-east-1` region; change `S3_REGION` inside the script if your setup differs.

When training is finished the results will end up in the `cloudless-data` bucket on S3, tarred and gzipped. You can download this and run it locally on your host against the test validation scripts to see how well training went. On your own machine _outside_ AWS run:

```
aws s3 ls cloudless-data
```

The result should be the latest GZIP file; if you are running several tests in parallel make sure to look at the private instance IP address in the GZIP file name to match it up with your particular run.

Run the following to generate validation graphs (with example GZIP filename), from your `cloudless` checked out directory _outside_ the AWS instance:

```
aws s3 cp s3://cloudless-data/caffe-results-12-16-15-host-ip-172-31-6-33-time-1450242072.tar.gz ~/tmp
gunzip ~/tmp/caffe-results-12-16-15-host-ip-172-31-6-33-time-1450242072.tar.gz
tar -xvf ~/tmp/caffe-results-12-16-15-host-ip-172-31-6-33-time-1450242072.tar
./src/cloudless/train/test.py --log_num 1 --note "Testing training run"
```

# Bounding Box/Inference System

This is the inference portion of the cloudless pipeline once you have trained a
model. It draw bounding boxes over cloud candidates. It's code lives in [src/cloudless/inference](src/cloudless/inference).

The primary script is `localization.py`, which generates candidate regions in an image using a [fork of Selective Search from here](https://github.com/BradNeuberg/selective_search_py). The [unforked Selective Search](https://github.com/belltailjp/selective_search_py) had a dependency on Python 3 but was back ported to Python 2.7 as part of the Cloudless work.

To set up, first make sure you've de-activated any virtualenv environment that might be running for the annotation tool; the bounding box system does not use virtualenv.

You must install the [Python 2.7 fork of Selective Search](https://github.com/BradNeuberg/selective_search_py) first, as well as Caffe obviously. Both CAFFE_HOME and SELECTIVE_SEARCH must be set to where these live as environment variables.

Example usage for generating bounding box regions for the example shown at the top of this README:

```
cd src/cloudless/inference
./localization.py -i ../../../examples/rapideye_cloud_2.jpg --classes cloud-classes.txt --config ../../caffe_model/bvlc_alexnet/bounding_box.prototxt --weights ../../caffe_model/bvlc_alexnet/bvlc_alexnet_finetuned.caffemodel --ks 1 --max_regions 600 --only_for_class 1 --platform gpu --threshold 9.0
open rapideye_cloud_3-regions.png
```

This will write out the image with bounding boxes drawn on it, including a JSON file with machine readable info on the top bounding boxes, such as rapideye_cloud_2.json, containing all the detected bounding boxes. This can be used by downstream code to ignore or eliminate these clouds, such as treating them as an alpha mask.

During development it is sometimes useful to test against the full, non-tuned version of ImageNet (not Cloudless) for debugging purposes. This is done against the full set of ImageNet classes:

```
cd src/cloudless/inference
./localization.py -i cat.jpg --classes imagenet-classes.txt --config ../../caffe_model/bvlc_alexnet/bounding_box_imagenet.prototxt --weights ../../caffe_model/bvlc_alexnet/bvlc_alexnet.caffemodel --ks 125 --max_regions 4
open cat-regions.png
```

import os

def determine_output_ending():
    """
    We add a unique ending to our output files so they can stack over time,
    such as output0001.log. This method determines an appropriate, unused
    ending, incrementing through those that are already present.
    """
    file_found = False
    idx = 1
    while not file_found:
        if not os.path.isfile(LOG_DIR + "/output%04d.png" % (idx)):
          return "%04d" % (idx)
        idx += 1

ROOT_DIR = "."
LOG_DIR = ROOT_DIR + "/logs"

# The number to append to output files, such as output0001.log.
OUTPUT_ENDING = determine_output_ending()

OUTPUT_LOG_PREFIX = LOG_DIR + "/output" + OUTPUT_ENDING

# Where to write out log files that aide in understanding how training went.
OUTPUT_LOG_PATH = OUTPUT_LOG_PREFIX + ".log"

# Where to write out training and validation result graphs.
OUTPUT_GRAPH_PATH = OUTPUT_LOG_PREFIX

CAFFE_HOME = os.environ.get("CAFFE_HOME")

MODEL_ROOT = ROOT_DIR + "/src/caffe_model/bvlc_alexnet"
SOLVER_FILE = MODEL_ROOT + "/solver.prototxt"
DEPLOY_FILE = MODEL_ROOT + "/deploy.prototxt"
WEIGHTS_NON_FINETUNED = MODEL_ROOT + "/bvlc_alexnet_orig.caffemodel"
WEIGHTS_FINETUNED = MODEL_ROOT + "/bvlc_alexnet_finetuned.caffemodel"

TRAINING_FILE = ROOT_DIR + "/data/leveldb/train_leveldb"
VALIDATION_FILE = ROOT_DIR + "/data/leveldb/validation_leveldb"

# Path to ImageNet's mean file, which AlexNet is trained on and which must be used as a mask.
TRAINING_MEAN_FILE = ROOT_DIR + "/data/imagenet/imagenet_mean.binaryproto"

# A pickled version of the ImageNet mean file, useful at deploy rather than train time.
TRAINING_MEAN_PICKLE = ROOT_DIR + "/data/imagenet/imagenet_mean.npy"

WIDTH = 256
HEIGHT = 256

# The width and height at inference time, which is different then at training time since
# we have clipping and transformation layers in Caffe.
INFERENCE_WIDTH = 227
INFERENCE_HEIGHT = 227

LANDSAT_ROOT = ROOT_DIR + "/data/landsat"
LANDSAT_IMAGES = LANDSAT_ROOT + "/images"
LANDSAT_METADATA = LANDSAT_ROOT + "/metadata/training-validation-set.csv"

PLANETLAB_ROOT = ROOT_DIR + "/data/planetlab"
# TODO(brad): Update the annotation image generation code to be able to dump the cropped images in
# one place and the metadata file in another.
PLANETLAB_UNBOUNDED_IMAGES = PLANETLAB_ROOT + "/metadata"
PLANETLAB_BOUNDED_IMAGES = PLANETLAB_ROOT + "/images/bounded"
PLANETLAB_METADATA = PLANETLAB_ROOT + "/metadata/annotated.json"

# Architecture string that will appear on graphs; good for relatively stable
# hyperparameter tuning.
ARCHITECTURE = "AlexNet fine tune - freeze convolution; Landsat data"

import glob
import os

import numpy as np
import caffe
import plyvel
import skimage
from caffe_pb2 import Datum

import constants

def predict(image_path):
    """
    Takes a single image, and makes a prediction whether it has a cloud or not.
    """

    print "Generating prediction for %s..." % image_path

    _initialize_caffe()
    im = caffe.io.load_image(image_path)
    prob = _predict_image(im)
    print "Probability this image has a cloud: {}%".format(prob)
    """
    Takes validation images and runs them through a trained model to see how
    well they do. Generates statistics like precision and recall, a confusion matrix,
    ROC curve, etc. in order to gauge progress.
def _initialize_caffe():
    """
    Initializes Caffe to prepare to run some data through the model for inference.
    """
    caffe.set_mode_gpu()
    net = caffe.Net(constants.DEPLOY_FILE, constants.WEIGHTS_FINETUNED, caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
    # PIL.Image loads the data with the channel last.
    transformer.set_transpose("data", (2, 0, 1))
    # Mean pixel.
    transformer.set_mean("data", np.load(constants.TRAINING_MEAN_PICKLE).mean(1).mean(1))
    # The reference model operates on images in [0, 255] range instead of [0, 1].
    transformer.set_raw_scale("data", 255)
    # The reference model has channels in BGR order instead of RGB.
    transformer.set_channel_swap("data", (2, 1, 0))

    # Deal with only a single image to predict.
    net.blobs["data"].reshape(1, 3, constants.INFERENCE_HEIGHT, constants.INFERENCE_WIDTH)
def _predict_image(im):
    """
    Given a caffe.io.load_image, returns the probability that it contains a cloud.
    """

    net.blobs["data"].data[...] = transformer.preprocess("data", im)
    out = net.forward()

    probs = out["prob"][0]
    prob_cloud = probs[0] * 100.0
    return prob_cloud

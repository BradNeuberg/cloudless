#!/usr/bin/env python
import os
import sys
import argparse
import time
import re
from operator import itemgetter

sys.path.append(os.environ.get('SELECTIVE_SEARCH'))

CAFFE_HOME = os.environ.get("CAFFE_HOME")
sys.path.append(CAFFE_HOME)

# Suppress annoying output from Caffe.
os.environ['GLOG_minloglevel'] = '1'

from selective_search import *
from skimage.transform import resize
import caffe
import numpy as np

def parse_command_line():
    parser = argparse.ArgumentParser(
      description="""Generate bounding boxes with classifications on an image.""")
    parser.add_argument(
        "-i",
        "--image",
        help="input image",
        default='cat.jpg'
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output image with bounding boxes",
        default='cat-regions.jpg'
    )
    parser.add_argument(
        "-m",
        "--dimension",
        help="image dimension of input to a trained classifier",
        default=(227, 227, 3)
    )
    parser.add_argument(
        "-P",
        "--pad",
        type=int,
        help="padding to use during cropping",
        default=16
    )
    parser.add_argument(
        "-c",
        "--config",
        help="prototxt for Caffe",
        default="alexnet.prototxt"
    )
    parser.add_argument(
        "-w",
        "--weights",
        help="weights for Caffe",
        default="alexnet.caffemodel"
    )
    parser.add_argument(
        "-p",
        "--platform",
        help="specify platform.",
        default="cpu"
    )
    parser.add_argument(
        "-l",
        "--classes",
        help="(optional) file with classes (format: 000 class)",
        default="imagenet-classes.txt"
    )
    parser.add_argument(
        "-r",
        "--regions",
        help="(optional) maximum number of bounding box regions to choose",
        type=int,
        default=3
    )
    parser.add_argument(
        "-t",
        "--threshold",
        help="(optional) percentage threshold of confidence necessary for a bounding box to be included",
        type=float,
        default=13.0
    )
    parser.add_argument(
        "-D",
        "--dump-regions",
        help="whether to dump cropped region candidates to files to aid debugging",
        action="store_true",
        default=True
    )

    args = parser.parse_args()

    if os.environ.get("SELECTIVE_SEARCH") == None:
        print("You must set SELECTIVE_SEARCH. Example:")
        print("export SELECTIVE_SEARCH=/usr/local/selective_search_py")
        exit(1)

    if os.environ.get("CAFFE_HOME") == None:
        print("You must set CAFFE_HOME to point to where Caffe is installed. Example:")
        print("export CAFFE_HOME=/usr/local/caffe")
        exit(1)

    return args

# Choose X number of regions that match some threshold.
# Take the original image and draw bounding boxes on them; add labels to the bounding boxes

def gen_regions(image, dims, pad):
    """
    Generates candidate regions for object detection using selective search.
    """

    print "Generating cropped regions..."
    assert(len(dims) == 3)
    img = skimage.io.imread(image)
    regions = selective_search(img, ks=[300])

    crops = []
    for conf, (x0, y0, x1, y1) in regions:
        if x0 - pad >= 0:
            x0 = x0 - pad
        if y0 - pad >= 0:
            y0 = y0 - pad
        if x1 + pad <= dims[0]:
            x1 = x1 + pad
        if y1 + pad <= dims[0]:
            y1 = y1 + pad
        region = img[x0:x1, y0:y1, :]
        candidate = resize(region, dims)
        crops.append((conf, candidate, region, (x0, y0, x1, y1)))

    print "Generated {} crops".format(len(crops))

    return crops

def get_region_filename(idx):
    """ Generates a region filename. """
    return "regions/%s.jpg" % idx

def dump_regions(crops):
    """ Writes out region proposals to the disk in regions/ for debugging. """
    if not os.path.exists("regions"):
        os.makedirs("regions")

    for idx, img in enumerate(crops):
        fname =  get_region_filename(idx)
        skimage.io.imsave(fname, img[2])

    print "Wrote regions out to disk in regions/"

def classify(images, config, weights):
    """ Classifies our region proposals. """
    print("Classifying: %d region images" % len(images))

    assert(os.path.isfile(config) and os.path.isfile(weights))

    # Caffe swaps RGB channels
    channel_swap = [2, 1, 0]

    # TODO: resizing on incoming config to make batching more efficient, predict
    # loops over each image, slow
    # Make classifier.
    classifier = caffe.Classifier(config,
                                  weights,
                                  raw_scale=255,
                                  channel_swap=channel_swap,
                                 )

    # Classify.
    return classifier.predict(images, oversample=False)

def load_classes(class_file):
    classes = {}
    if os.path.isfile(class_file):
        f = open(class_file, 'r')
        for line in f: # '001 goldfish'
            key = int(line.split(" ")[0])
            value = line.split(" ",1)[1].strip('\n')
            classes[key] = value

    return classes

def sort_predictions(classes, predictions, bboxes):
    """ Sorts predictions from most probable to least, generate extra metadata about them. """
    results = []
    for idx, pred in enumerate(predictions):
        results.append({
            "class": classes[np.argmax(pred)],
            "prob": pred[np.argmax(pred)],
            "fname": get_region_filename(idx),
            "coords": bboxes[idx],
        })
    results.sort(key=itemgetter("prob"), reverse=True)

    print_predictions(classes, results)

    return results

def print_predictions(classes, predictions):
    """ Prints out the predictions for debugging. """
    for idx, pred in enumerate(predictions):
        print("prob: {}, class: {}, file: {}, coords: {}".format(
            predictions[idx]["prob"],
            predictions[idx]["class"],
            predictions[idx]["fname"],
            predictions[idx]["coords"],
        ))

def main(argv):
    args = parse_command_line()
    crops = gen_regions(args.image, args.dimension, args.pad)

    if args.dump_regions:
        dump_regions(crops)

    images = [entry[1] for entry in crops]
    classes = load_classes(args.classes)
    predictions = classify(images, args.config, args.weights)

    bboxes = [entry[3] for entry in crops]
    predictions = sort_predictions(classes, predictions, bboxes)


if __name__ == '__main__':
    main(sys.argv)

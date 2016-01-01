#!/usr/bin/env python
import os
import shutil
import sys
import argparse
import time
import re
import random
from decimal import Decimal
from operator import itemgetter
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.environ.get('SELECTIVE_SEARCH'))

CAFFE_HOME = os.environ.get("CAFFE_HOME")
sys.path.append(CAFFE_HOME)

# Suppress annoying output from Caffe.
os.environ['GLOG_minloglevel'] = '1'

from selective_search import *
import features
from skimage.transform import resize
import caffe
import numpy as np
import simplejson as json

# TODO: It looks like PNG images aren't working, only JPG images.

def parse_command_line():
    parser = argparse.ArgumentParser(
      description="""Generate bounding boxes with classifications on an image.""")
    parser.add_argument(
        "-i",
        "--image",
        help="input image",
        default="cat.jpg"
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
        "--max_regions",
        help="(optional) maximum number of bounding box regions to choose",
        type=int,
        default=3
    )
    parser.add_argument(
        "-t",
        "--threshold",
        help="(optional) percentage threshold of confidence necessary for a bounding box to be included",
        type=float,
        default=10.0
    )
    parser.add_argument(
        "-D",
        "--dump-regions",
        help="whether to dump cropped region candidates to files to aid debugging",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "-k",
        "--ks",
        help="value for the ks argument controlling selective search region formation",
        type=int,
        default=100
    )
    parser.add_argument(
        "--only_for_class",
        help="""only draw bounding boxes for regions that match some class; draws bounding boxes
             for any class found if not given""",
        type=int,
        default=None
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

def gen_regions(image, dims, pad, ks):
    """
    Generates candidate regions for object detection using selective search.
    """

    print "Generating cropped regions..."
    assert(len(dims) == 3)
    regions = selective_search(image, ks=[ks], feature_masks=[features.SimilarityMask(
        size=1,
        color=1,
        texture=1,
        fill=1,
    )])

    crops = []
    for conf, (y0, x0, y1, x1) in regions:
        if x0 - pad >= 0:
            x0 = x0 - pad
        if y0 - pad >= 0:
            y0 = y0 - pad
        if x1 + pad <= dims[0]:
            x1 = x1 + pad
        if y1 + pad <= dims[0]:
            y1 = y1 + pad
        # Images are rows, then columns, then channels.
        region = image[y0:y1, x0:x1, :]
        candidate = resize(region, dims)
        crops.append((conf, candidate, region, (x0, y0, x1, y1)))

    print "Generated {} crops".format(len(crops))

    return crops

def get_region_filename(idx):
    """ Generates a region filename. """
    return "bbox-regions/%s.jpg" % idx

def dump_regions(crops):
    """ Writes out region proposals to the disk in regions/ for debugging. """
    shutil.rmtree("bbox-regions", ignore_errors=True)
    os.makedirs("bbox-regions")

    for idx, img in enumerate(crops):
        fname =  get_region_filename(idx)
        skimage.io.imsave(fname, img[2])

    print "Wrote regions out to disk in bbox-regions/"

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
            "class_idx": np.argmax(pred),
            "class": classes[np.argmax(pred)],
            "prob": pred[np.argmax(pred)],
            "fname": get_region_filename(idx),
            "coords": bboxes[idx],
        })
    results.sort(key=itemgetter("prob"), reverse=True)

    return results

def filter_predictions(predictions, max_regions, threshold):
    """
    Filters predictions down to just those that are above or equal to a certain threshold, with
    a max number of results controlled by 'max_regions'.
    """
    results = [entry for entry in predictions if entry["prob"] >= threshold]
    results = results[0:max_regions]
    return results

def print_predictions(classes, predictions):
    """ Prints out the predictions for debugging. """
    print "Top predictions:"
    for idx, pred in enumerate(predictions):
        print("prob: {}, class: {}, file: {}, coords: {}".format(
            predictions[idx]["prob"],
            predictions[idx]["class"],
            predictions[idx]["fname"],
            predictions[idx]["coords"],
        ))

def draw_bounding_boxes(image_path, image, classes, predictions, only_for_class=None):
    image = Image.fromarray(numpy.uint8(image))
    dr = ImageDraw.Draw(image, "RGBA")

    colors = {}
    for idx, pred in enumerate(predictions):
        x0, y0, x1, y1 = pred["coords"]

        color = (255, 255, 0, 60)
        # If we want to display multiple classes, randomly generate a color for it.
        if not only_for_class:
            class_idx = pred["class_idx"]
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if class_idx in colors:
                color = colors[class_idx]
            colors[class_idx] = color

        dr.rectangle(((x0, y0), (x1, y1)), fill=color)

        if not only_for_class:
            dr.text((x0, y0), pred["class"], fill=color)

    filename = os.path.splitext(image_path)[0] + "-regions.png"
    image.save(filename)

    print "Image with drawn bounding boxes saved to %s" % filename

def dump_bounding_box_info(image_path, predictions):
    """ Writes out our top predictions to a JSON file for other tools to work with. """
    filename = os.path.splitext(image_path)[0] + "-regions.json"
    # Make sure we can serialize our Python float values.
    for entry in predictions:
        entry["prob"] = Decimal("%.7g" % entry["prob"])

    with open(filename, "w") as f:
        f.write(json.dumps(predictions, use_decimal=True, indent=4, separators=(',', ': ')))

    print "Bounding box info saved as JSON to %s" % filename

def main(argv):
    args = parse_command_line()
    image_path = os.path.abspath(args.image)
    image = skimage.io.imread(image_path)

    crops = gen_regions(image, args.dimension, args.pad, args.ks)

    if args.dump_regions:
        dump_regions(crops)

    images = [entry[1] for entry in crops]
    classes = load_classes(args.classes)
    config = os.path.abspath(args.config)
    weights = os.path.abspath(args.weights)
    predictions = classify(images, config, weights)

    bboxes = [entry[3] for entry in crops]
    predictions = sort_predictions(classes, predictions, bboxes)
    predictions = filter_predictions(predictions, args.max_regions, args.threshold)
    print_predictions(classes, predictions)

    draw_bounding_boxes(image_path, image, classes, predictions, args.only_for_class)
    dump_bounding_box_info(image_path, predictions)

if __name__ == '__main__':
    main(sys.argv)

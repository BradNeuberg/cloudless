#!/usr/bin/env python

import argparse
import shutil
import os
import time
import csv
import json
import random

from PIL import (Image, ImageOps)
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import plyvel
from caffe_pb2 import Datum

import utils

def parse_command_line():
    parser = argparse.ArgumentParser(description="""Prepares data for training via Caffe""")
    parser.add_argument("--input_metadata", help="Path to where our bounding box metadata is",
        type=str, default="data/planetlab/metadata/annotated.json")
    parser.add_argument("--input_images", help="Path to where our unbounded images are",
        type=str, default="data/planetlab/metadata")
    parser.add_argument("--output_images", help="Path to place our cropped, bounded images",
        type=str, default="data/planetlab/images/bounded")
    parser.add_argument("--output_leveldb", help="Path to place our prepared leveldb directories",
        type=str, default="data/leveldb")
    parser.add_argument("--width", help="Width of image at training time (it will be scaled to this)",
        type=int, default=256)
    parser.add_argument("--height", help="Height of image at training time (it will be scaled to this)",
        type=int, default=256)
    parser.add_argument("--log_path", help="The path to where to place log files",
        type=str, default="logs")
    parser.add_argument("--log_num", help="""Number that will be appended to log files; this will
        be automatically padded and added with zeros, such as output00001.log""", type=int,
        default=1)
    parser.add_argument("--do_augmentation", help="Whether to do data augmentation",
        dest="do_augmentation", action="store_true")

    parser.set_defaults(do_augmentation=False)
    args = vars(parser.parse_args())

    utils.assert_caffe_setup()

    # Ensure the random number generator always starts from the same place for consistent tests.
    random.seed(0)

    log_path = os.path.abspath(args["log_path"])
    log_num = args["log_num"]
    (output_ending, output_log_prefix, output_log_file) = utils.get_log_path_details(log_path, log_num)

    input_metadata = os.path.abspath(args["input_metadata"])
    input_images = os.path.abspath(args["input_images"])
    output_images = os.path.abspath(args["output_images"])
    output_leveldb = os.path.abspath(args["output_leveldb"])
    prepare_data(input_metadata, input_images, output_images, output_leveldb, args["width"],
        args["height"], args["do_augmentation"], output_log_prefix)

def prepare_data(input_metadata, input_images, output_images, output_leveldb, width, height,
                 do_augmentation, output_log_prefix):
    """
    Prepares our training and validation data sets for use by Caffe.
    """
    print "Preparing data..."

    print "\tParsing Planet Labs data into independent cropped bounding boxes using %s..." % input_metadata
    details = _crop_planetlab_images(_get_planetlab_details(input_metadata, input_images), output_images)

    train_paths, validation_paths, train_targets, validation_targets = _split_data_sets(details)

    if do_augmentation == True:
        print "\tDoing data augmentation..."
        train_paths, train_targets = _do_augmentation(output_images, train_paths, train_targets)
    else:
        print "\tNot doing data augmentation"

    # TODO(brad): Balance classes if command-line option provided to do so.
    #_balance_classes(details)

    _print_input_details(details, train_paths, train_targets, output_log_prefix, do_augmentation)

    print "\tSaving prepared data..."
    training_file = os.path.join(output_leveldb, "train_leveldb")
    validation_file = os.path.join(output_leveldb, "validation_leveldb")
    _generate_leveldb(training_file, train_paths, train_targets, width, height)
    _generate_leveldb(validation_file, validation_paths, validation_targets, width, height)

    _copy_validation_images(validation_paths, output_images)

def _get_planetlab_details(input_metadata, input_images):
    """
    Loads available image paths and image filenames for planetlab, along with any bounding boxes
    that might be present for clouds in them.
    """
    print "Using the following metadata file: %s" % input_metadata

    with open(input_metadata) as data_file:
        details = json.load(data_file)

    for entry in details:
        entry["image_path"] = os.path.join(input_images, entry["image_name"])
        entry["target"] = 0
        if len(entry["image_annotation"]):
            entry["target"] = 1

        bboxes = []
        for bbox in entry["image_annotation"]:
            bbox = bbox.split(",")
            x = int(bbox[0])
            y = int(bbox[1])
            width = int(bbox[2])
            height = int(bbox[3])
            bboxes.append({
                "left": x,
                "upper": y,
                "right": x + width,
                "lower": y + height
            })
        entry["image_annotation"] = bboxes

    return details

# The first early iteration of the system used Landsat data to confirm the pipeline; left here
# commented out for future reference.
# def _get_landsat_details():
#     """
#     Loads available image paths and image filenames for landsat, along with their target values if
#     they contain clouds or not (1 if there is a cloud, 0 otherwise).
#     """
#
#     LANDSAT_ROOT = ROOT_DIR + "/data/landsat"
#     LANDSAT_IMAGES = LANDSAT_ROOT + "/images"
#     LANDSAT_METADATA = LANDSAT_ROOT + "/metadata/training-validation-set.csv"
#
#     image_paths = []
#     targets = []
#     with open(LANDSAT_METADATA, 'r') as csvfile:
#       entryreader = csv.reader(csvfile, delimiter=',', quotechar='"')
#       firstline = True
#       for row in entryreader:
#         if firstline:
#             firstline = False
#             continue
#         filename = row[0]
#         has_cloud = 0
#         if row[1] == "1":
#           has_cloud = 1
#
#         image_paths.append(os.path.join(LANDSAT_IMAGES, filename))
#         targets.append(has_cloud)
#
#     return {
#         "image_paths": image_paths,
#         "targets": targets,
#     }

def _crop_planetlab_images(details, output_images):
    """
    Generates cropped cloud and non-cloud images from our annotated bounding boxes, dumping
    them into the file system and returning their full image paths with whether they are targets
    or not.
    """
    image_paths = []
    targets = []
    raw_input_images_count = 0

    # Remove the directory to ensure we don't get old data runs included.
    shutil.rmtree(output_images, ignore_errors=True)
    os.makedirs(output_images)

    for entry in details:
        raw_input_images_count = raw_input_images_count + 1
        if entry["target"] == 0:
            # Nothing to crop, but remove the alpha channel.
            new_path = os.path.join(output_images, entry["image_name"])

            im = Image.open(entry["image_path"])
            im = _rgba_to_rgb(im)
            im.save(new_path)

            image_paths.append(new_path)
            targets.append(entry["target"])
            print "\t\tProcessed non-cloud image %s" % new_path
        elif entry["target"] == 1:
            (root, ext) = os.path.splitext(entry["image_name"])

            cloud_num = 1
            for bbox in entry["image_annotation"]:
                im = Image.open(entry["image_path"])
                try:
                    new_path = os.path.join(output_images, "%s_cloud_%03d%s" % (root, cloud_num, ext))

                    new_im = im.crop((bbox["left"], bbox["upper"], bbox["right"], bbox["lower"]))
                    new_im = _rgba_to_rgb(new_im)
                    new_im.save(new_path)

                    image_paths.append(new_path)
                    targets.append(1)

                    print "\t\tProcessed cloud cropped image %s" % new_path

                    cloud_num += 1
                except:
                    print "\t\tInvalid crop value: {}".format(bbox)

    return {
        "image_paths": image_paths,
        "targets": targets,
        "raw_input_images_count": raw_input_images_count,
    }

def _print_input_details(details, train_paths, train_targets, output_log_prefix, do_augmentation):
    """
    Prints out statistics about our input data.
    """
    positive_cloud_class = 0
    negative_cloud_class = 0
    for entry in train_targets:
        if entry == 1:
            positive_cloud_class = positive_cloud_class + 1
        else:
            negative_cloud_class = negative_cloud_class + 1

    ratio = min(float(positive_cloud_class), float(negative_cloud_class)) / \
            max(float(positive_cloud_class), float(negative_cloud_class))

    statistics = """\t\tInput data details during data preparation:
        \t\tTotal # of raw input images for training/validation: %d
        \t\tTotal # of generated bounding box images for training/validation: %d
        \t\tPositive cloud count (# of images with clouds) in training data: %d
        \t\tNegative cloud count (# of images without clouds) in training data: %d
        \t\tRatio: %.2f
        \t\tTotal # of input images including data augmentation: %d
        \t\tBalanced classes: no
        \t\tData augmentation: %r
        \t\tAdding inference bounding boxes into training data: no""" \
        % ( \
            details["raw_input_images_count"],
            len(details["image_paths"]),
            positive_cloud_class,
            negative_cloud_class,
            ratio,
            len(train_paths),
            do_augmentation,
    )
    print statistics

    statistics_log_file = output_log_prefix + ".preparation_statistics.txt"
    print "\t\tSaving preparation statistics to %s..." % statistics_log_file
    with open(statistics_log_file, "w") as f:
        f.write(statistics)

# def _balance_classes(details):
#     """
#     Ensures we have the same number of positive and negative cloud/not cloud classes.
#     """

def _split_data_sets(details):
    """
    Shuffles and splits our datasets into training and validation sets.
    """
    image_paths = details["image_paths"]
    targets = details["targets"]

    print "\tShuffling data..."
    (image_paths, targets) = shuffle(image_paths, targets, random_state=0)

    print "\tSplitting data 80% training, 20% validation..."
    return train_test_split(image_paths, targets, train_size=0.8, test_size=0.2, \
      random_state=0)

def _copy_validation_images(validation_paths, output_images):
    """
    Takes bounded validation images and copies them to a separate directory so we can distinguish
    training from validation images later on.
    """
    validation_images = os.path.join(output_images, "validation")
    shutil.rmtree(validation_images, ignore_errors=True)
    os.makedirs(validation_images)
    print "\tCopying validation images to %s..." % validation_images
    for i in xrange(len(validation_paths)):
        old_path = validation_paths[i]
        filename = os.path.basename(old_path)
        new_path = os.path.join(validation_images, filename)
        shutil.copyfile(old_path, new_path)

# TODO: We really should be doing this at training time instead as on-demand transformations, via a
# Python-based layer right after input data is loaded. Example:
# https://github.com/BVLC/caffe/blob/master/python/caffe/test/test_python_layer.py
def _do_augmentation(output_images, train_paths, train_targets):
    """
    Augments our training data through cropping, rotations, and mirroring.
    """
    result_train_paths = []
    result_train_targets = []

    augmentation_dir = os.path.join(output_images, "augmentation")
    shutil.rmtree(augmentation_dir, ignore_errors=True)
    os.makedirs(augmentation_dir)

    # Note: our Caffe train_val.prototxt already does mirroring and basic cropping, so just
    # do 90 degree rotations.

    for i in xrange(len(train_paths)):
        input_path = train_paths[i]
        input_target = train_targets[i]

        print "\t\tDoing data augmentation for %s" % input_path
        try:
            im = Image.open(input_path)
            (width, height) = im.size
            process_me = []

            result_train_paths.append(input_path)
            result_train_targets.append(input_target)

            # Only crop if our image is above some size or else it gets nonsensical.
            process_me.append(im)
            # if width >= 100 and height >= 100:
            #     _crop_image(im, width, height, process_me)

            # Now rotate all of these four ways 90 degrees and then mirror them.
            process_me = _rotate_images(process_me)
            #process_me = _mirror_images(process_me)

            # Note: the original image is the first entry. Remove it since its already saved to disk
            # so we don't accidentally duplicate it again.
            del process_me[0]

            _, base_filename = os.path.split(input_path)
            base_filename, file_extension = os.path.splitext(base_filename)
            for idx in xrange(len(process_me)):
                entry = process_me[idx]
                new_path = os.path.join(augmentation_dir,
                    base_filename + "_augment_" + str(idx + 1) + file_extension)
                result_train_paths.append(new_path)
                result_train_targets.append(input_target)

                entry.save(new_path)
        except:
            print "\t\tWarning: Unable to work with %s" % input_path

    return result_train_paths, result_train_targets

def _crop_image(im, width, height, process_me):
    """
    Crops an image into its four corners and its center, adding them to process_me.
    """
    process_me.append(im.crop((0, 0, width / 2, height / 2)))
    process_me.append(im.crop((width / 2, 0, width, height / 2)))
    process_me.append(im.crop((0, height / 2, width / 2, height)))
    process_me.append(im.crop((width / 2, height / 2, width, height)))

    # Crop the center.
    center_width = width / 2
    center_height = height / 2
    center_left = (width - center_width) / 2
    center_top = (height - center_height) / 2
    center_right = (width + center_width) / 2
    center_bottom = (height + center_height) / 2
    process_me.append(im.crop((center_left, center_top, center_right, center_bottom)))

def _rotate_images(process_me):
    """
    Rotates the images given in process_me by all four possible 90 degrees.
    """
    results = []
    for orig_im in process_me:
        results.append(orig_im)
        rotated_im = orig_im
        for i in range(3):
            rotated_im = rotated_im.rotate(90)
            results.append(rotated_im)
    return results

def _mirror_images(process_me):
    """
    Mirrors the given images horizontally.
    """
    results = []
    for orig_im in process_me:
        results.append(orig_im)
        results.append(ImageOps.mirror(orig_im))
    return results

def _generate_leveldb(file_path, image_paths, targets, width, height):
    """
    Caffe uses the LevelDB format to efficiently load its training and validation data; this method
    writes paired out faces in an efficient way into this format.
    """
    print "\t\tGenerating LevelDB file at %s..." % file_path
    shutil.rmtree(file_path, ignore_errors=True)
    db = plyvel.DB(file_path, create_if_missing=True)
    wb = db.write_batch()
    commit_every = 10000
    start_time = int(round(time.time() * 1000))
    for idx in range(len(image_paths)):
      # Each image is a top level key with a keyname like 00000000011, in increasing
      # order starting from 00000000000.
      key = utils.get_key(idx)

      # Do common normalization that might happen across both testing and validation.
      try:
        image = _preprocess_data(_load_numpy_image(image_paths[idx], width, height))
      except:
        print "\t\t\tWarning: Unable to process leveldb image %s" % image_paths[idx]
        continue

      # Each entry in the leveldb is a Caffe protobuffer "Datum" object containing details.
      datum = Datum()
      datum.channels = 3 # RGB
      datum.height = height
      datum.width = width
      datum.data = image.tostring()
      datum.label = targets[idx]
      value = datum.SerializeToString()
      wb.put(key, value)

      if (idx + 1) % commit_every == 0:
        wb.write()
        del wb
        wb = db.write_batch()
        end_time = int(round(time.time() * 1000))
        total_time = end_time - start_time
        print "\t\t\tWrote batch, key: %s, time for batch: %d ms" % (key, total_time)
        start_time = int(round(time.time() * 1000))

    end_time = int(round(time.time() * 1000))
    total_time = end_time - start_time
    print "\t\t\tWriting final batch, time for batch: %d ms" % total_time
    wb.write()
    db.close()

def _preprocess_data(data):
    """
    Applies any standard preprocessing we might do on data, whether it is during
    training or testing time. 'data' is a numpy array of unrolled pixel vectors with
    a remote sensing image.
    """
    # Do nothing for now.
    # We don't scale it's values to be between 0 and 1 as our Caffe model will do that.
    return data

def _load_numpy_image(image_path, width, height):
    """
    Turns one of our testing image paths into an actual image, converted into a numpy array.
    """
    im = Image.open(image_path)
    # Scale the image to the size required by our neural network.
    im = im.resize((width, height))
    data = np.asarray(im)
    data = np.reshape(data, (3, height, width))
    return data

def _rgba_to_rgb(im):
    """
    Drops the alpha channel in an RGB image.
    """
    return im.convert("RGB")

if __name__ == "__main__":
    parse_command_line()

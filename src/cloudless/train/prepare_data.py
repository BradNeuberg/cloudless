import shutil
import os
import time
import csv
import json

from PIL import Image
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import plyvel
from caffe_pb2 import Datum

import constants
import utils

def prepare_data():
    """
    Prepares our training and validation data sets for use by Caffe.
    """
    print "Preparing data..."

    print "\tParsing Planet Labs data into independent cropped bounding boxes..."
    #details = _get_landsat_details()
    details = _crop_planetlab_images(_get_planetlab_details())

    print "\t\tClass details before balancing (balancing not implemented yet):"
    _print_input_details(details)

    # TODO(brad): Balance classes.
    #_balance_classes(details)

    (train_paths, validation_paths, train_targets, validation_targets) = _split_data_sets(details)

    print "\tSaving prepared data..."
    _generate_leveldb(constants.TRAINING_FILE, train_paths, train_targets)
    _generate_leveldb(constants.VALIDATION_FILE, validation_paths, validation_targets)

def _get_planetlab_details():
    """
    Loads available image paths and image filenames for landsat, along with any bounding boxes
    that might be present for clouds in them.
    """
    print "location: %s" % constants.PLANETLAB_METADATA

    with open(constants.PLANETLAB_METADATA) as data_file:
        details = json.load(data_file)

    for entry in details:
        entry["image_path"] = os.path.join(constants.PLANETLAB_UNBOUNDED_IMAGES,
            entry["image_name"])
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

def _get_landsat_details():
    """
    Loads available image paths and image filenames for landsat, along with their target values if
    they contain clouds or not (1 if there is a cloud, 0 otherwise).
    """

    image_paths = []
    targets = []
    with open(constants.LANDSAT_METADATA, 'r') as csvfile:
      entryreader = csv.reader(csvfile, delimiter=',', quotechar='"')
      firstline = True
      for row in entryreader:
        if firstline:
            firstline = False
            continue
        filename = row[0]
        has_cloud = 0
        if row[1] == "1":
          has_cloud = 1

        image_paths.append(os.path.join(constants.LANDSAT_IMAGES, filename))
        targets.append(has_cloud)

    return {
        "image_paths": image_paths,
        "targets": targets,
    }

def _crop_planetlab_images(details):
    """
    Generates cropped cloud and non-cloud images from our annotated bounding boxes, dumping
    them into the file system and returning their full image paths with whether they are targets
    or not.
    """

    image_paths = []
    targets = []

    # Remove the directory to ensure we don't get old data runs included.
    shutil.rmtree(constants.PLANETLAB_BOUNDED_IMAGES, ignore_errors=True)
    os.makedirs(constants.PLANETLAB_BOUNDED_IMAGES)

    for entry in details:
        if entry["target"] == 0:
            # Nothing to crop, but remove the alpha channel.
            new_path = os.path.join(constants.PLANETLAB_BOUNDED_IMAGES, entry["image_name"])

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
                try:
                    new_path = os.path.join(constants.PLANETLAB_BOUNDED_IMAGES,
                        "%s_cloud_%03d%s" % (root, cloud_num, ext))

                    im = Image.open(entry["image_path"])
                    im = im.crop((bbox["left"], bbox["upper"], bbox["right"], bbox["lower"]))
                    im = _rgba_to_rgb(im)
                    im.save(new_path)

                    image_paths.append(new_path)
                    targets.append(1)

                    print "\t\tProcessed cloud cropped image %s" % new_path

                    cloud_num += 1
                except:
                    # TODO(brad): Modify the annotation UI to not be able to produce invalid
                    # crop values.
                    print "\t\tInvalid crop value"

    return {
        "image_paths": image_paths,
        "targets": targets,
    }

def _print_input_details(details):
    """
    Prints out statistics about our input data.
    """

    positive_cloud_class = 0
    negative_cloud_class = 0
    for entry in details["targets"]:
        if entry == 1:
            positive_cloud_class = positive_cloud_class + 1
        else:
            negative_cloud_class = negative_cloud_class + 1

    print "\t\tInput data details:"
    print "\t\t\tTotal number of input images: %d" % len(details["image_paths"])
    print "\t\t\tPositive cloud count (# of images with clouds): %d" % positive_cloud_class
    print "\t\t\tNegative cloud count (# of images without clouds): %d" % negative_cloud_class
    # TODO(brad): Print out ratio of positive to negative.

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

def _generate_leveldb(file_path, image_paths, targets):
    """
    Caffe uses the LevelDB format to efficiently load its training and validation data; this method
    writes paired out faces in an efficient way into this format.
    """
    print "\t\tGenerating LevelDB file at %s..." % file_path
    shutil.rmtree(file_path, ignore_errors=True)
    db = plyvel.DB(file_path, create_if_missing=True)
    wb = db.write_batch()
    commit_every = 250000
    start_time = int(round(time.time() * 1000))
    for idx in range(len(image_paths)):
      # Each image is a top level key with a keyname like 00000000011, in increasing
      # order starting from 00000000000.
      key = utils.get_key(idx)

      # Do common normalization that might happen across both testing and validation.
      image = _preprocess_data(_load_numpy_image(image_paths[idx]))

      # Each entry in the leveldb is a Caffe protobuffer "Datum" object containing details.
      datum = Datum()
      datum.channels = 3 # RGB
      datum.height = constants.HEIGHT
      datum.width = constants.WIDTH
      # TODO: Should I swap the color channels to BGR?
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
    # TODO(neuberg): Confirm that the AlexNet proto file has correct scaling values
    # for the kinds of pixels we will use.

    return data

def _load_numpy_image(image_path):
    """
    Turns one of our testing image paths into an actual image, converted into a numpy array.
    """

    im = Image.open(image_path)
    # Scale the image to the size required by our neural network.
    im = im.resize((constants.WIDTH, constants.HEIGHT))
    data = np.asarray(im)
    data = np.reshape(data, (3, constants.HEIGHT, constants.WIDTH))
    return data

def _rgba_to_rgb(im):
    """
    Drops the alpha channel in an RGB image.
    """
    return im.convert('RGB')

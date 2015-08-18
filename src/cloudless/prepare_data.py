import shutil
import os
import glob
import time
import csv

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

    details = _get_landsat_details()
    _print_input_details(details)
    (train_paths, validation_paths, train_targets, validation_targets) = _split_data_sets(details)

    print "Saving prepared data..."
    _generate_leveldb(constants.TRAINING_FILE, train_paths, train_targets)
    _generate_leveldb(constants.VALIDATION_FILE, validation_paths, validation_targets)

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

  print "\tInput data details:"
  print "\t\tTotal number of input images: %d" % len(details["image_paths"])
  print "\t\tPositive cloud count (# of images with clouds): %d" % positive_cloud_class
  print "\t\tNegative cloud count (# of images without clouds): %d" % negative_cloud_class

def _split_data_sets(details):
  """
  Splits our datasets into training and validation sets.
  """

  print "Splitting data 80% training, 20% validation..."
  return train_test_split(details["image_paths"], details["targets"], train_size=0.8, test_size=0.2, \
      random_state=0)

def _generate_leveldb(file_path, image_paths, targets):
    """
    Caffe uses the LevelDB format to efficiently load its training and validation data; this method
    writes paired out faces in an efficient way into this format.
    """
    print "\tGenerating LevelDB file at %s..." % file_path
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
        print "\t\tWrote batch, key: %s, time for batch: %d ms" % (key, total_time)
        start_time = int(round(time.time() * 1000))

    end_time = int(round(time.time() * 1000))
    total_time = end_time - start_time
    print "\t\tWriting final batch, time for batch: %d ms" % total_time
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
    im.thumbnail((constants.WIDTH, constants.HEIGHT), Image.ANTIALIAS)
    data = np.asarray(im)
    data = np.reshape(data, (3, constants.HEIGHT, constants.WIDTH))
    return data

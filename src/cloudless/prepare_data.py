import shutil
import time

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

    # TODO: Generate a mean normalization file for Caffe.
    # TODO: Take the raw images and whether they have clouds or not, split them into
    # training and validation test sets, and pack them into training and validation
    # LevelDB databases.

def _generate_leveldb(self, file_path, image, target, single_data):
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
    for idx in range(len(pairs)):
      # Each image is a top level key with a keyname like 00000000011, in increasing
      # order starting from 00000000000.
      key = utils.get_key(idx)

      # Do things like mean normalize, etc. that happen across both testing and validation.
      paired_image = self._preprocess_data(paired_image)

      # Each entry in the leveldb is a Caffe protobuffer "Datum" object containing details.
      datum = Datum()
      # TODO(neuberg): Confirm that this is the correct way to setup RGB images for
      # Caffe for our dataset.
      datum.channels = 3
      datum.height = constants.HEIGHT
      datum.width = constants.WIDTH
      datum.data = image.tostring()
      datum.label = target[idx]
      value = datum.SerializeToString()
      wb.put(key, value)

      if (idx + 1) % commit_every == 0:
        wb.write()
        del wb
        wb = db.write_batch()
        end_time = int(round(time.time() * 1000))
        total_time = end_time - start_time
        print "Wrote batch, key: %s, time for batch: %d ms" % (key, total_time)
        start_time = int(round(time.time() * 1000))

    wb.write()
    db.close()

def _preprocess_data(self, data):
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

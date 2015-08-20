#!/usr/bin/env python2
import os, sys, argparse, glob, time, re, re
CAFFE_HOME = os.environ.get("CAFFE_HOME")
sys.path.append(CAFFE_HOME)

# suppress annoying output from Caffe
os.environ['GLOG_minloglevel'] = '1'

import caffe
import numpy as np
from pprint import pprint

def numericalSort(value):
  numbers = re.compile(r'(\d+)')
  parts = numbers.split(value)
  parts[1::2] = map(int, parts[1::2])
  return parts

def parse_command_line():
  parser = argparse.ArgumentParser(description="Inference script for Caffe")
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
      help="Specify platform.",
      default="cpu"
  )
  parser.add_argument(
      "-i",
      "--input",
      help="Input (image|directory)",
      default="cat.jpg"
  )
  parser.add_argument(
      "-e",
      "--ext",
      help="Image extension",
      default="jpg"
  )
  parser.add_argument(
      "-l",
      "--classes",
      help="(optional) File with classes (format: 000 class)",
      default="imagenet-classes.txt"
  )
  args = parser.parse_args()

  if os.environ.get("CAFFE_HOME") == None:
    print("You must set CAFFE_HOME to point to where Caffe is installed. Example:")
    print("export CAFFE_HOME=/usr/local/caffe")
    exit(1)

  return args

def main(argv):
  """
  Inference script for Caffe useful for recognition tasks.
  Input: image (or folder)
  Output: classifies all images in folder
  """
  args = parse_command_line()

  if args.platform == "gpu":
    caffe.set_mode_gpu()

  classes = {}
  if os.path.isfile(args.classes):
    f = open(args.classes, 'r')
    for line in f: # '001 goldfish'
      key = int(line.split(" ")[0])
      value = line.split(" ",1)[1].strip('\n')
      classes[key] = value

  fnames = []
  if os.path.isdir(args.input):
    images = sorted(glob.glob(args.input + "/*." + args.ext), key=numericalSort)
    assert (len(images) > 0)

    inp = []
    for fn in images:
      img = caffe.io.load_image(fn)
      inp.append(img)
      fnames.append(fn)
  else:
    inp = [caffe.io.load_image(args.input)]
    fnames.append(args.input)

  print("Classifying: %d images" % len(inp))

  assert(os.path.isfile(args.config) and os.path.isfile(args.weights))

  # Caffe swaps RGB channels
  channel_swap = [2, 1, 0]

  # TODO: resizing on incoming config to make batching more efficient, predict
  # loops over each image, slow
  # Make classifier.
  classifier = caffe.Classifier(args.config,
                                args.weights,
                                raw_scale=255,
                                channel_swap=channel_swap,
                               )

  # Classify.
  predictions = classifier.predict(inp, oversample=False)
  if classes:
    for idx,pred in enumerate(predictions):
      print("{}: {}".format(fnames[idx], classes[int(pred)]))
  else:
    print(predictions)

if __name__ == '__main__':
    main(sys.argv)

#!/usr/bin/env python2
import os, sys, argparse, glob, time
CAFFE_HOME = os.environ.get("CAFFE_HOME")
sys.path.append(CAFFE_HOME)

# suppress annoying output from Caffe
os.environ['GLOG_minloglevel'] = '1'

import caffe
import numpy as np

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

  if os.path.isdir(args.input):
    images = glob.glob(args.input + "/*." + args.ext)
    assert (len(images) > 0)

    inp = []
    for fn in images:
      img = caffe.io.load_image(fn)
      inp.append(img)
  else:
    inp = [caffe.io.load_image(args.input)]

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
  print(predictions)

if __name__ == '__main__':
    main(sys.argv)

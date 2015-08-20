#!/usr/bin/env python3

import sys, time, os, argparse
sys.path.append(os.environ.get('SELECTIVE_SEARCH'))

from selective_search import *
from skimage.transform import resize
import numpy as np

def parse_command_line():
  parser = argparse.ArgumentParser(
      description="""Generate Regions using selective search.""")
  parser.add_argument("-i", "--image", help="input image", default='cat.jpg')
  parser.add_argument("-o", "--output", help="output directory",
                      default='./regions')
  parser.add_argument("-m", "--dimension", help="image dimension to resize crops to",
                      default=(227,227,3))

  args = parser.parse_args()

  if os.environ.get("SELECTIVE_SEARCH") == None:
    print("You must set SELECTIVE_SEARCH. Example:")
    print("export SELECTIVE_SEARCH=/usr/local/selective_search_py")
    exit(1)

  return args

def gen_regions(image, dims):
  """
  Generates candidate regions for object detection using selective search
  """

  assert(len(dims) == 3)
  img = skimage.io.imread(image)
  start = time.time()
  regions = selective_search(img, ks=[100])
  print("Selective search time: %.2f [s]" % (time.time() - start))

  resize_t = 0
  crops = []
  for conf, (x0, y0, x1, y1) in regions:
    region = img[x0:x1, y0:y1, :]
    st = time.time()
    candidate = resize(region, dims)
    resize_t += time.time() - st
    crops.append( (conf, candidate, region) )

  print("Resize time: %.2f [s]" % resize_t)

  return crops

def main(argv):
  args = parse_command_line()
  if not os.path.exists(args.output):
    os.makedirs(args.output)

  crops = gen_regions(args.image, args.dimension)

  # write out
  for idx, img in enumerate(crops):
    fname = args.output + '/%s.jpg' % idx
    skimage.io.imsave(fname, img[1])

  print("Crops generated: %d" % idx)

if __name__ == '__main__':
  main(sys.argv)

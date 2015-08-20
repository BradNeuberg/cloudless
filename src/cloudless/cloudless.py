#!/usr/bin/env python
import argparse
import os
import random

import train.constants as constants
import train.predict as predict
from train.prepare_data import prepare_data
from train.train import train

def parse_command_line():
    parser = argparse.ArgumentParser(
        description="""Train, validate, and test a classifier that will detect clouds in
        remote sensing data.""")
    parser.add_argument("-p", "--prepare-data", help="Prepare training and validation data.",
        action="store_true")
    parser.add_argument("-t", "--train", help="""Train classifier. Use --graph to generate quality
        graphs""", action="store_true")
    parser.add_argument("-g", "--graph", help="Generate training graphs.", action="store_true")
    parser.add_argument("--predict", help="""Make a prediction about a single image.
        Provide path to image.""", nargs=1, type=str)
    parser.add_argument("--note", help="Adds extra note onto generated quality graph.", type=str)

    args = vars(parser.parse_args())

    if os.environ.get("CAFFE_HOME") == None:
        print "You must set CAFFE_HOME to point to where Caffe is installed. Example:"
        print "export CAFFE_HOME=/usr/local/caffe"
        exit(1)

    # Ensure the random number generator always starts from the same place for consistent tests.
    random.seed(0)

    if args["prepare_data"] == True:
        prepare_data()
    if args["train"] == True:
        train(args["graph"], note=args["note"])
    if args["predict"] != None:
        image_path = args["predict"][0]
        predict.predict(image_path)

if __name__ == "__main__":
    parse_command_line()

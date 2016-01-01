#!/usr/bin/env python

import argparse
import shutil
import subprocess
import sys
import fileinput
import re
import random
import os

import utils

def parse_command_line():
    parser = argparse.ArgumentParser(description="Trains Caffe model against prepared data")
    parser.add_argument("--log_path", help="The path to where to place log files",
        type=str, default="logs")
    parser.add_argument("--log_num", help="""Number that will be appended to log files; this will
        be automatically padded and added with zeros, such as output00001.log""", type=int,
        default=1)
    parser.add_argument("--note", help="Adds extra note into training logs.", type=str,
        default=None)
    parser.add_argument("--solver", help="The path to our Caffe solver prototxt file",
        type=str, default="src/caffe_model/bvlc_alexnet/solver.prototxt")
    parser.add_argument("--input_weight_file", help="""A pre-trained Caffe model that we will use
        to start training with in order to fine-tune from""", type=str,
        default="src/caffe_model/bvlc_alexnet/bvlc_alexnet.caffemodel")
    parser.add_argument("--output_weight_file", help="""Where to place the final, trained Caffe
        model""", type=str, default="logs/latest_bvlc_alexnet_finetuned.caffemodel")

    args = vars(parser.parse_args())

    caffe_home = utils.assert_caffe_setup()

    # Ensure the random number generator always starts from the same place for consistent tests.
    random.seed(0)

    log_path = os.path.abspath(args["log_path"])
    log_num = args["log_num"]
    (output_ending, output_log_prefix, output_log_file) = utils.get_log_path_details(log_path, log_num)

    solver = os.path.abspath(args["solver"])
    input_weight_file = os.path.abspath(args["input_weight_file"])
    output_weight_file = os.path.abspath(args["output_weight_file"])
    train(caffe_home, log_path, output_log_file, solver, input_weight_file, output_weight_file,
        args["note"])

def train(caffe_home, log_path, output_log_file, solver, input_weight_file, output_weight_file, note):
    """ Trains Caffe finetuning the given model. """
    print("Training using data")

    _run_trainer(caffe_home, log_path, output_log_file, solver, input_weight_file, note)

    _generate_parsed_logs(caffe_home, log_path, output_log_file)
    (training_details, validation_details) = utils.parse_logs(log_path, output_log_file)
    _move_trained_weight_file(log_path, output_log_file, output_weight_file)

    print "Finished training!"

def _run_trainer(caffe_home, log_path, output_log_file, solver, input_weight_file, note):
    """
    Runs Caffe to train the model.
    """
    print("\tRunning trainer...")
    with open(output_log_file, "w") as f:
        process = subprocess.Popen([caffe_home + "/build/tools/caffe", "train",
            "--solver=" + solver, "--weights=" + input_weight_file],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        if note != None:
            sys.stdout.write("Details for this training run: {}\n".format(note))

        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            f.write(line)

        print("\t\tTraining output saved to %s" % output_log_file)

def _generate_parsed_logs(caffe_home, log_path, output_log_file):
    """
    Takes the raw Caffe output created while training the model in order
    to generate reduced statistics, such as giving iterations vs. test loss.
    """

    print("\tParsing logs...")
    process = subprocess.Popen([caffe_home + "/tools/extra/parse_log.py",
        output_log_file, log_path], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)

    shutil.rmtree(output_log_file + ".validate", ignore_errors=True)
    shutil.move(output_log_file + ".test", output_log_file + ".validate")

    # Convert the commas in the files into tabs to make them easier to read.
    log_files = [output_log_file + ".train", output_log_file + ".validate"]
    for line in fileinput.input(log_files, inplace=True):
        line = line.replace(u",", u"\t")
        if fileinput.isfirstline():
            # HACK(neuberg): The column headers with tabs don't quite line up, so shorten
            # some column names and add a tab.
            line = line.replace(u"NumIters", u"Iters")
            line = line.replace(u"LearningRate", u"\tLR")

        sys.stdout.write(line)
    fileinput.close()

    logs = [
        {"title": "Testing", "filename": "train"},
        {"title": "Validation", "filename": "validate"}
    ]
    for log in logs:
        print("\n\t\tParsed %s log:" % log["title"])
        with open(output_log_file + "." + log["filename"], "r") as f:
            lines = f.read().split("\n")
            for line in lines:
                print("\t\t\t%s" % line)

    print("\t\tParsed training log saved to %s" % (output_log_file + ".train"))
    print("\t\tParsed validation log saved to %s\n" % (output_log_file + ".validate"))

def _move_trained_weight_file(log_path, output_log_file, output_weight_file):
    """
    Moves our trained weight file somewhere else.
    """
    trained_weight_file = _get_trained_weight_file(log_path, output_log_file)
    print "\tMoving trained weight file from %s to %s" % (trained_weight_file, output_weight_file)
    shutil.copyfile(trained_weight_file, output_weight_file)

def _get_trained_weight_file(log_path, output_log_file):
    """
    Parses out the file name of the model weight file we just trained.
    """
    trained_weight_file = None
    with open(output_log_file) as f:
        content = f.read()
        # Note: not all versions of Caffe include the phrase 'binary proto file ' in the log output.
        trained_weight_file = re.findall("Snapshotting to (?:binary proto file )?(.*)$", content,
                re.MULTILINE)[-1]

    return trained_weight_file

if __name__ == "__main__":
    parse_command_line()

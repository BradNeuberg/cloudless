#!/usr/bin/env python
import argparse
import re
import random
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.font_manager import FontProperties

import utils
import predict

def parse_command_line():
    parser = argparse.ArgumentParser(description="""Tests a trained Caffe model to see how well
        it does, generating quality graphs and statistics""")
    parser.add_argument("--log_path", help="The path to where to place log files and graphs",
        type=str, default="logs")
    parser.add_argument("--log_num", help="""Number that will be appended to log files; this will
        be automatically padded and added with zeros, such as output00001.log""", type=int,
        default=1)
    parser.add_argument("--input_weight_file", help="""The trained and fine-tuned Caffe model that
        we will be testing; defaults to the last trained model from train.py""", type=str,
        default="logs/latest_bvlc_alexnet_finetuned.caffemodel")
    parser.add_argument("--note", help="Adds extra note onto generated quality graphs.", type=str,
        default="")
    parser.add_argument("--solver", help="The path to our Caffe solver prototxt file",
        type=str, default="src/caffe_model/bvlc_alexnet/solver.prototxt")
    parser.add_argument("--deploy", help="""Path to our Caffe deploy/inference time prototxt file""",
        type=str, default="src/caffe_model/bvlc_alexnet/deploy.prototxt")
    parser.add_argument("--threshold", help="""The percentage threshold over which we assume
        something is a cloud. Note that this value is from 0.0 to 100.0""", type=float, default=0.1)
    parser.add_argument("--validation_leveldb", help="""Path to where the validation leveldb file is""",
        type=str, default="data/leveldb/validation_leveldb")
    parser.add_argument("--width", help="Width of image during training", type=int, default=256)
    parser.add_argument("--height", help="Height of image during training", type=int, default=256)
    parser.add_argument("--inference_width", help="Width of image during training", type=int,
        default=227)
    parser.add_argument("--inference_height", help="Height of image during training", type=int,
        default=227)
    parser.add_argument("--training_mean_pickle", help="Path to pickled mean values", type=str,
        default="data/imagenet/imagenet_mean.npy")

    args = vars(parser.parse_args())

    print "Testing trained model..."

    caffe_home = utils.assert_caffe_setup()

    # Ensure the random number generator always starts from the same place for consistent tests.
    random.seed(0)

    log_path = os.path.abspath(args["log_path"])
    log_num = args["log_num"]
    (output_ending, output_log_prefix, output_log_file) = utils.get_log_path_details(log_path, log_num)
    output_graph_path = output_log_prefix

    (training_details, validation_details) = utils.parse_logs(log_path, output_log_file)

    plot_results(training_details, validation_details, args["note"], output_graph_path, args["solver"])
    validation_leveldb = os.path.abspath(args["validation_leveldb"])
    deploy = os.path.abspath(args["deploy"])
    input_weight_file = os.path.abspath(args["input_weight_file"])
    training_mean_pickle = os.path.abspath(args["training_mean_pickle"])
    predict.test_validation(args["threshold"], output_log_prefix, validation_leveldb,
        deploy, args["width"], args["height"], args["inference_width"],
        args["inference_height"], input_weight_file, training_mean_pickle)

def plot_results(training_details, validation_details, note, output_graph_path, solver):
    """
    Generates training/validation graphs.
    """

    _plot_loss(training_details, validation_details, note, output_graph_path, solver)
    _plot_accuracy(training_details, validation_details, note, output_graph_path, solver)

def _plot_loss(training_details, validation_details, note, output_graph_path, solver):
    """
    Plots training/validation loss side by side.
    """
    print "\tPlotting training/validation loss..."
    fig, ax1 = plt.subplots()
    ax1.plot(training_details["iters"], training_details["loss"], "b-")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Training Loss", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    ax2.plot(validation_details["iters"], validation_details["loss"], "r-")
    ax2.set_ylabel("Validation Loss", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    plt.suptitle("Iterations vs. Training/Validation Loss", fontsize=14)
    plt.title(_get_hyperparameter_details(note, solver), style="italic", fontsize=12)

    filename = output_graph_path + ".loss.png"
    plt.savefig(filename)
    plt.close()
    print("\t\tGraph saved to %s" % filename)

def _plot_accuracy(training_details, validation_details, note, output_graph_path, solver):
    """
    Plots training/validation accuracy over iterations.
    """
    print "\tPlotting training/validation accuracy..."

    fmt = '%.1f%%'
    yticks = mtick.FormatStrFormatter(fmt)

    fig, ax1 = plt.subplots()
    training_percentage = [percent * 100 for percent in training_details["accuracy"]]
    ax1.plot(training_details["iters"], training_percentage, "b-")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Training Accuracy", color="b")
    ax1.yaxis.set_major_formatter(yticks)
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    validation_percentage = [percent * 100 for percent in validation_details["accuracy"]]
    ax2.plot(validation_details["iters"], validation_percentage, "r-")
    ax2.set_ylabel("Validation Accuracy", color="r")
    ax2.yaxis.set_major_formatter(yticks)
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    plt.suptitle("Iterations vs. Training/Validation Accuracy", fontsize=14)
    plt.title(_get_hyperparameter_details(note, solver), style="italic", fontsize=12)

    filename = output_graph_path + ".accuracy.png"
    plt.savefig(filename)
    plt.close()
    print("\t\tGraph saved to %s" % filename)

def _get_hyperparameter_details(note, solver):
    """
    Parse out some of the values we need from the Caffe solver prototext file.
    """
    solver = open(solver, "r")
    details = solver.read()
    lr = re.search("^base_lr:\s*([0-9.]+)$", details, re.MULTILINE).group(1)
    max_iter = re.search("^max_iter:\s*([0-9.]+)$", details, re.MULTILINE).group(1)
    results = "(lr: %s; max_iter: %s" % (lr, max_iter)

    # Add any extra details into the graph if someone specified that on the command line.
    if note:
        results += "; %s" % note
    results += ")"
    return results

if __name__ == "__main__":
    parse_command_line()

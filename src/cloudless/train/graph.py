import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib.font_manager import FontProperties

import constants

def plot_results(training_details, validation_details, note=None):
    """
    Generates training/validation graphs.
    """

    _plot_loss(training_details, validation_details, note)
    _plot_accuracy(training_details, validation_details, note)

def _plot_loss(training_details, validation_details, note=None):
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
    plt.title(_get_hyperparameter_details(note), style="italic", fontsize=12)

    filename = constants.OUTPUT_GRAPH_PATH + ".loss.png"
    plt.savefig(filename)
    plt.close()
    print("\t\tGraph saved to %s" % filename)

def _plot_accuracy(training_details, validation_details, note=None):
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
    plt.title(_get_hyperparameter_details(note), style="italic", fontsize=12)

    filename = constants.OUTPUT_GRAPH_PATH + ".accuracy.png"
    plt.savefig(filename)
    plt.close()
    print("\t\tGraph saved to %s" % filename)

def _get_hyperparameter_details(note=None):
    """
    Parse out some of the values we need from the Caffe solver prototext file.
    """
    solver = open(constants.SOLVER_FILE, "r")
    details = solver.read()
    lr = re.search("^base_lr:\s*([0-9.]+)$", details, re.MULTILINE).group(1)
    max_iter = re.search("^max_iter:\s*([0-9.]+)$", details, re.MULTILINE).group(1)
    results = "(lr: %s; max_iter: %s; %s" % (lr, max_iter, constants.ARCHITECTURE)

    # Add any extra details into the graph if someone specified that on the command line.
    if note:
        results += "; %s" % note
    results += ")"
    return results

import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

import constants

def plot_results(training_details, validation_details, note=None):
    """
    Generates a combined training/validation graph.
    """
    print "\tPlotting results..."
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

    legend_font = FontProperties()
    legend_font.set_size("small")
    blue_line = mpatches.Patch(color="blue", label="Training Loss")
    red_line = mpatches.Patch(color="red", label="Validation Loss")
    plt.legend(handles=[blue_line, red_line], prop=legend_font, loc="lower right")

    plt.suptitle("Iterations vs. Training/Validation Loss", fontsize=14)
    plt.title(get_hyperparameter_details(note), style="italic", fontsize=12)

    plt.savefig(constants.OUTPUT_GRAPH_PATH)
    print("\t\tGraph saved to %s" % constants.OUTPUT_GRAPH_PATH)

def get_hyperparameter_details(note=None):
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

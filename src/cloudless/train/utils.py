import csv
import re
import os

def get_key(idx):
    """
    Each image is a top level key with a keyname like 00059999, in increasing
    order starting from 00000000.
    """
    return "%08d" % (idx,)

def assert_caffe_setup():
    """
    Makes sure that Caffe's environment CAFFE_HOME variable is set. If so, returns its value.
    Otherwise ends execution of this script.
    """
    caffe_home = os.environ.get("CAFFE_HOME")
    if caffe_home == None:
        print "You must set CAFFE_HOME to point to where Caffe is installed. Example:"
        print "export CAFFE_HOME=/usr/local/caffe"
        exit(1)
    return caffe_home

def get_log_path_details(log_path, log_num):
    """
    Generates path information on to where to put our log files.
    """
    output_ending = "%04d" % (log_num)
    output_log_prefix = os.path.join(log_path, "output" + output_ending)
    output_log_file = output_log_prefix + ".log"
    return (output_ending, output_log_prefix, output_log_file)

def parse_logs(log_path, output_log_file):
    """
    Parses our training and validation logs produced from a Caffe training run in order to return
    them in a way we can work with.
    """
    training_iters = []
    training_loss = []
    training_accuracy = []
    for line in csv.reader(open(output_log_file + ".train"), delimiter="\t", skipinitialspace=True):
        if re.search("Iters", str(line)):
            continue

        training_iters.append(int(float(line[0])))
        training_accuracy.append(float(line[3]))
        training_loss.append(float(line[4]))

    validation_iters = []
    validation_loss = []
    validation_accuracy = []
    for line in csv.reader(open(output_log_file + ".validate"), delimiter="\t",
                            skipinitialspace=True):
        if re.search("Iters", str(line)):
            continue

        validation_iters.append(int(float(line[0])))
        validation_accuracy.append(float(line[3]))
        validation_loss.append(float(line[4]))

    return (
        {
            "iters": training_iters,
            "loss": training_loss,
            "accuracy": training_accuracy
        }, {
            "iters": validation_iters,
            "loss": validation_loss,
            "accuracy": validation_accuracy
        }
    )

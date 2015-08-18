import subprocess

import constants

def train(output_graphs, data=None, weight_file=None, note=None):
    print("Training data, generating graphs: %r" % output_graphs)

    run_trainer()

def run_trainer():
    """
    Runs Caffe to train the model.
    """
    print("\tRunning trainer...")
    with open(constants.OUTPUT_LOG_PATH, "w") as f:
        process = subprocess.Popen([constants.CAFFE_HOME + "/build/tools/caffe", "train",
            "--solver=" + constants.SOLVER_FILE],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            f.write(line)

        print("\t\tTraining output saved to %s" % constants.OUTPUT_LOG_PATH)


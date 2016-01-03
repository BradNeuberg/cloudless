#!/usr/bin/env python
import argparse
import os

# Suppress annoying output from Caffe.
os.environ['GLOG_minloglevel'] = '1'

import caffe
import numpy as np
import plyvel
import skimage
from caffe_pb2 import Datum

def parse_command_line():
    parser = argparse.ArgumentParser(description="""Predicts for a single image using the trained
        model whether it has a cloud or not""")
    parser.add_argument("--image", help="""The image to attempt to predict""", default="")
    parser.add_argument("--deploy", help="""Path to our Caffe deploy/inference time prototxt file""",
        type=str, default="src/caffe_model/bvlc_alexnet/deploy.prototxt")
    parser.add_argument("--input_weight_file", help="""The trained and fine-tuned Caffe model that
        we will be testing; defaults to the last trained model from train.py""", type=str,
        default="logs/latest_bvlc_alexnet_finetuned.caffemodel")
    parser.add_argument("--training_mean_pickle", help="Path to pickled mean values", type=str,
        default="data/imagenet/imagenet_mean.npy")
    parser.add_argument("--inference_width", help="Width of image during training", type=int,
        default=227)
    parser.add_argument("--inference_height", help="Height of image during training", type=int,
        default=227)

    args = vars(parser.parse_args())

    image = os.path.abspath(args["image"])
    deploy = os.path.abspath(args["deploy"])
    input_weight_file = os.path.abspath(args["input_weight_file"])
    training_mean_pickle = os.path.abspath(args["os.path.abspath"])
    predict(image, deploy, input_weight_file, training_mean_pickle, args["inference_width"],
        args["inference_height"])

def predict(image_path, deploy_file, input_weight_file, training_mean_pickle, inference_width,
        inference_height):
    """
    Takes a single image, and makes a prediction whether it has a cloud or not.
    """

    print "Generating prediction for %s..." % image_path

    net, transformer = _initialize_caffe(deploy_file, input_weight_file, training_mean_pickle,
            inference_width, inference_height)
    im = caffe.io.load_image(image_path)
    prob = _predict_image(im, net, transformer)
    print "Probability this image has a cloud: {0:.2f}%".format(prob)

def test_validation(threshold, output_log_prefix, validation_leveldb, deploy_file, width, height,
            inference_width, inference_height, input_weight_file, training_mean_pickle):
    """
    Takes validation images and runs them through a trained model to see how
    well they do. Generates statistics like precision and recall, F1, and a confusion matrix,
    in order to gauge progress.
    """
    print "Generating predictions for validation images..."

    validation_data = _load_validation_data(validation_leveldb, width, height)
    target_details = _run_through_caffe(validation_data, deploy_file, input_weight_file, threshold,
            training_mean_pickle, inference_width, inference_height)
    statistics = _calculate_positives_negatives(target_details)

    accuracy = _calculate_accuracy(statistics)
    precision = _calculate_precision(statistics)
    recall = _calculate_recall(statistics)
    f1 = _calculate_f1(precision, recall)

    # TODO: Write these out to a file as well as the screen.
    results = ""
    results += "\n"
    results += "\nStatistics on validation dataset using threshold %f:" % threshold
    results += "\n\tAccuracy: {0:.2f}%".format(accuracy)
    results += "\n\tPrecision: %.2f" % precision
    results += "\n\tRecall: %.2f" % recall
    results += "\n\tF1 Score: %.2f" % f1

    results += "\n"
    results += _print_confusion_matrix(statistics)

    print results

    with open(output_log_prefix + ".statistics.txt", "w") as f:
        f.write(results)

def _load_validation_data(validation_leveldb, width, height):
    """
    Loads all of our validation data from our leveldb database, producing unrolled numpy input
    vectors ready to test along with their correct, expected target values.
    """

    print "\tLoading validation data..."
    input_vectors = []
    expected_targets = []

    db = plyvel.DB(validation_leveldb)
    for key, value in db:
        datum = Datum()
        datum.ParseFromString(value)

        data = np.fromstring(datum.data, dtype=np.uint8)
        data = np.reshape(data, (3, height, width))
        # Move the color channel to the end to match what Caffe wants.
        data = np.swapaxes(data, 0, 2) # Swap channel with width.
        data = np.swapaxes(data, 0, 1) # Swap width with height, to yield final h x w x channel.

        input_vectors.append(data)
        expected_targets.append(datum.label)

    db.close()

    print "\t\tValidation data has %d images" % len(input_vectors)

    return {
        "input_vectors": np.asarray(input_vectors),
        "expected_targets": np.asarray(expected_targets)
    }

def _initialize_caffe(deploy_file, input_weight_file, training_mean_pickle, inference_width,
            inference_height):
    """
    Initializes Caffe to prepare to run some data through the model for inference.
    """
    caffe.set_mode_gpu()
    net = caffe.Net(deploy_file, input_weight_file, caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
    # PIL.Image loads the data with the channel last.
    transformer.set_transpose("data", (2, 0, 1))
    # Mean pixel.
    transformer.set_mean("data", np.load(training_mean_pickle).mean(1).mean(1))
    # The reference model operates on images in [0, 255] range instead of [0, 1].
    transformer.set_raw_scale("data", 255)
    # The reference model has channels in BGR order instead of RGB.
    transformer.set_channel_swap("data", (2, 1, 0))

    net.blobs["data"].reshape(1, 3, inference_height, inference_width)

    return (net, transformer)

def _run_through_caffe(validation_data, deploy_file, input_weight_file, threshold,
            training_mean_pickle, inference_width, inference_height):
    """
    Runs our validation images through Caffe.
    """

    print "\tInitializing Caffe..."
    net, transformer = _initialize_caffe(deploy_file, input_weight_file, training_mean_pickle,
            inference_width, inference_height)

    print "\tComputing probabilities using Caffe..."
    results = []
    for idx in range(len(validation_data["input_vectors"])):
        im = validation_data["input_vectors"][idx]
        prob = _predict_image(im, net, transformer)
        expected_target = validation_data["expected_targets"][idx]
        predicted_target = 0
        if prob >= threshold:
            predicted_target = 1
        results.append({
            "expected_target": expected_target,
            "predicted_target": predicted_target
        })

    return results

def _predict_image(im, net, transformer):
    """
    Given a caffe.io.load_image, returns the probability that it contains a cloud.
    """

    net.blobs["data"].data[...] = transformer.preprocess("data", im)
    out = net.forward()

    probs = out["prob"][0]
    prob_cloud = probs[1] * 100.0
    return prob_cloud

def _calculate_positives_negatives(target_details):
    """
    Takes expected and actual target values, generating true and false positives and negatives,
    including the actual correct # of positive and negative values.
    """

    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0
    actual_positive = 0
    actual_negative = 0
    for idx in range(len(target_details)):
        predicted_target = target_details[idx]["predicted_target"]
        expected_target = target_details[idx]["expected_target"]

        if expected_target == 1:
            actual_positive = actual_positive + 1
        else:
            actual_negative = actual_negative + 1

        if predicted_target == 1 and expected_target == 1:
            true_positive = true_positive + 1
        elif predicted_target == 0 and expected_target == 0:
            true_negative = true_negative + 1
        elif predicted_target == 1 and expected_target == 0:
            false_positive = false_positive + 1
        elif predicted_target == 0 and expected_target == 1:
            false_negative = false_negative + 1

    return {
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "actual_positive": float(actual_positive),

        "true_negative": float(true_negative),
        "false_negative": float(false_negative),
        "actual_negative": float(actual_negative),
    }

def _calculate_accuracy(s):
    top = (s["true_positive"] + s["true_negative"])
    bottom = (s["actual_positive"] + s["actual_negative"])
    return (top / bottom) * 100.0

def _calculate_precision(s):
    return s["true_positive"] / (s["true_positive"] + s["false_positive"])

def _calculate_recall(s):
    return s["true_positive"] / (s["true_positive"] + s["false_negative"])

def _calculate_f1(precision, recall):
    return 2.0 * ((precision * recall) / (precision + recall))

def _print_confusion_matrix(s):
    results = ""
    results += "\nConfusion matrix:"
    results += "\n\t\t\t\tPositive\t\tNegative"
    results += "\nPositive (%d)\t\t\tTrue Positive (%d)\tFalse Positive (%d)" % \
        (s["actual_positive"], s["true_positive"], s["false_positive"])
    results += "\nNegative (%d)\t\t\tFalse Negative (%d)\tTrue Negative (%d)" % \
        (s["actual_negative"], s["false_negative"], s["true_negative"])
    return results

if __name__ == "__main__":
    parse_command_line()

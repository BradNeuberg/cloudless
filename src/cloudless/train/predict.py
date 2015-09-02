import os

import caffe
import numpy as np
import plyvel
import skimage
from caffe_pb2 import Datum

import constants

def predict(image_path):
    """
    Takes a single image, and makes a prediction whether it has a cloud or not.
    """

    print "Generating prediction for %s..." % image_path

    net, transformer = _initialize_caffe()
    im = caffe.io.load_image(image_path)
    prob = _predict_image(im, net, transformer)
    print "Probability this image has a cloud: {0:.2f}%".format(prob)

def test_validation():
    """
    Takes validation images and runs them through a trained model to see how
    well they do. Generates statistics like precision and recall, F1, and a confusion matrix,
    in order to gauge progress.
    """
    print "Generating predictions for validation images..."

    validation_data = _load_validation_data()
    target_details = _run_through_caffe(validation_data)
    statistics = _calculate_positives_negatives(target_details)

    accuracy = _calculate_accuracy(statistics)
    precision = _calculate_precision(statistics)
    recall = _calculate_recall(statistics)
    f1 = _calculate_f1(precision, recall)

    # TODO: Write these out to a file as well as the screen.
    results = ""
    results += "\n"
    results += "\nStatistics on validation dataset:"
    results += "\n\tAccuracy: {0:.2f}%".format(accuracy)
    results += "\n\tPrecision: %.2f" % precision
    results += "\n\tRecall: %.2f" % recall
    results += "\n\tF1 Score: %.2f" % f1

    results += "\n"
    results += _print_confusion_matrix(statistics)

    print results

    with open(constants.OUTPUT_LOG_PREFIX + ".statistics.txt", "w") as f:
        f.write(results)

def _load_validation_data():
    """
    Loads all of our validation data from our leveldb database, producing unrolled numpy input
    vectors ready to test along with their correct, expected target values.
    """

    print "\tLoading validation data..."
    input_vectors = []
    expected_targets = []

    db = plyvel.DB(constants.VALIDATION_FILE)
    for key, value in db:
        datum = Datum()
        datum.ParseFromString(value)

        data = np.fromstring(datum.data, dtype=np.uint8)
        data = np.reshape(data, (3, constants.HEIGHT, constants.WIDTH))
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

def _initialize_caffe():
    """
    Initializes Caffe to prepare to run some data through the model for inference.
    """
    caffe.set_mode_gpu()
    net = caffe.Net(constants.DEPLOY_FILE, constants.WEIGHTS_FINETUNED, caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
    # PIL.Image loads the data with the channel last.
    # TODO: Think through whether these should be BGR during training and validation.
    transformer.set_transpose("data", (2, 0, 1))
    # Mean pixel.
    transformer.set_mean("data", np.load(constants.TRAINING_MEAN_PICKLE).mean(1).mean(1))
    # The reference model operates on images in [0, 255] range instead of [0, 1].
    transformer.set_raw_scale("data", 255)
    # The reference model has channels in BGR order instead of RGB.
    transformer.set_channel_swap("data", (2, 1, 0))

    net.blobs["data"].reshape(1, 3, constants.INFERENCE_HEIGHT, constants.INFERENCE_WIDTH)

    return (net, transformer)

def _run_through_caffe(validation_data):
    """
    Runs our validation images through Caffe.
    """

    print "\tInitializing Caffe..."
    net, transformer = _initialize_caffe()

    print "\tComputing probabilities using Caffe..."
    results = []
    for idx in range(len(validation_data["input_vectors"])):
        im = validation_data["input_vectors"][idx]
        prob = _predict_image(im, net, transformer)
        expected_target = validation_data["expected_targets"][idx]
        predicted_target = 0
        if prob >= constants.THRESHOLD:
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

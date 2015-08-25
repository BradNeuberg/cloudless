import glob
import os

import numpy as np
import caffe
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
    print "Probability this image has a cloud: {}%".format(prob)

def test_validation():
    """
    Takes validation images and runs them through a trained model to see how
    well they do. Generates statistics like precision and recall, F1, and a confusion matrix,
    in order to gauge progress.
    """
    print "Testing trained model against validation data..."

    validation_data = _load_validation_data()
    results = _run_through_caffe(validation_data)

    # TODO: Write these out to a file as well as the screen.
    # statistics = _calculate_positives_negatives(results)

    # accuracy = _calculate_accuracy(statistics)
    # precision = _calculate_precision(statistics)
    # recall = _calculate_recall(statistics)
    # f1 = _calculate_f1(statistics)

    # print "Accuracy: %f" % accuracy
    # print "Precision: %f" % precision
    # print "Recall: %f" % recall
    # print "F1 Score: %f" % f1_score

    # _draw_confusion_matrix(statistics)

def _load_validation_data():
    """
    Loads all of our validation data from our leveldb database, producing unrolled numpy input
    vectors ready to test along with their correct, expected target values.
    """

    print "\tLoading validation data..."
    input_vectors = []
    expected_targets = []

    # TODO: Change this back to validation_file.
    db = plyvel.DB(constants.TRAINING_FILE)
    for key, value in db:
        datum = Datum()
        datum.ParseFromString(value)

        # TODO: Make sure this works.
        data = np.fromstring(datum.data, dtype=np.uint8)
        data = np.reshape(data, (3, constants.HEIGHT, constants.WIDTH))
        data = skimage.img_as_float(data).astype(np.float32)

        input_vectors.append(data)
        expected_targets.append(datum.label)

    db.close()

    print "\tValidation data has %d images" % len(input_vectors)

    return {
        "input_vectors": np.array(input_vectors),
        "expected_targets": np.array(expected_targets)
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
    transformer.set_transpose("data", (2, 0, 1))
    # Mean pixel.
    transformer.set_mean("data", np.load(constants.TRAINING_MEAN_PICKLE).mean(1).mean(1))
    # The reference model operates on images in [0, 255] range instead of [0, 1].
    transformer.set_raw_scale("data", 255)
    # The reference model has channels in BGR order instead of RGB.
    transformer.set_channel_swap("data", (2, 1, 0))

    # Deal with only a single image to predict.
    net.blobs["data"].reshape(1, 3, constants.INFERENCE_HEIGHT, constants.INFERENCE_WIDTH)

    return (net, transformer)

def _run_through_caffe(validation_data):
    """
    Runs our validation images through Caffe.
    """
    # TODO: Run through all the results using Caffe, then return their actual values.

def _predict_image(im, net, transformer):
    """
    Given a caffe.io.load_image, returns the probability that it contains a cloud.
    """

    net.blobs["data"].data[...] = transformer.preprocess("data", im)
    out = net.forward()

    probs = out["prob"][0]
    prob_cloud = probs[0] * 100.0
    return prob_cloud

def _calculate_positives_negatives(target_details):
    """
    Takes expected and actual target values, generating true and false positives and negatives,
    including the actual correct # of positive and negative values.
    """

    # TODO

    return {
        "positive": {
            "true": float(true_positive),
            "false": float(false_positive),
            "actual": float(actual_positive)
        },
        "negative": {
            "true": float(true_negative),
            "false": float(false_negative),
            "actual": float(actual_negative)
        }
    }

# def _calculate_accuracy(statistics):
#     # TODO

# def _calculate_precision(statistics):
#     # TODO

# def _calculate_recall(statistics):
#     # TODO

# def _calculate_f1(statistics):
#     # TODO

# def _generate_confusion_matrix(statistics):
#     # TODO

import idx2numpy
import numpy as np

"""This file takes the MNIST dataset files
and converts the data to a 28x28 numpy array (matrix) that the network can use.
"""


training_images_raw = "./data/MNIST_dataset/train-images.idx3-ubyte"
training_labels_raw = "./data/MNIST_dataset/train-labels.idx1-ubyte"


training_images = idx2numpy.convert_from_file(training_images_raw).copy()
training_labels = idx2numpy.convert_from_file(training_labels_raw).copy()


test_images_raw = "./data/MNIST_dataset/test-images.idx3-ubyte"
test_labels_raw = "./data/MNIST_dataset/test-labels.idx1-ubyte"

test_images = idx2numpy.convert_from_file(test_images_raw).copy()
test_labels = idx2numpy.convert_from_file(test_labels_raw).copy()


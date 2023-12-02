import idx2numpy
import gzip
import numpy as np

"""This file takes the MNIST dataset files
and converts the data to a 28x28 numpy array (matrix) that the network can use.
"""
def extract_images(filename, num_images, image_width):
    print("Retrieving images from", filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(image_width * image_width * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, image_width, image_width)
        return data

def extract_labels(filename, num_images):
    print("Retrieving labels from", filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

fashion = False

if fashion: dataset = "Fashion_MNIST_dataset"
else: dataset = "MNIST_dataset"


training_images_file = f"./data/{dataset}/train-images-idx3-ubyte.gz"
training_labels_file = f"./data/{dataset}/train-labels-idx1-ubyte.gz"

training_images = extract_images(training_images_file, 60_000, 28)
training_labels = extract_labels(training_labels_file, 60_000)


test_images_file = f"./data/{dataset}/t10k-images-idx3-ubyte.gz"
test_labels_file = f"./data/{dataset}/t10k-labels-idx1-ubyte.gz"


test_images = extract_images(test_images_file, 10_000, 28)
test_labels = extract_labels(test_labels_file, 10_000)

valid_nums = [np.random.randint(0, len(test_images)) for _ in range(150)]
validation_images = [test_images[i] for i in valid_nums]
validation_labels = [test_labels[i] for i in valid_nums]

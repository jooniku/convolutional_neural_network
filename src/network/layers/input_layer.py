from src.mnist_data_processor import training_images, training_labels, validation_images, validation_labels, test_images, test_labels
import numpy as np


class InputLayer:
    """This class takes in data in correct, preprocessed format.

    Args:
        NeuralNetwork (_type_): _description_
    """

    def __init__(self):
        pass

    def normalize_images(self, images):
        images = np.array(images)
        images -= int(np.mean(images))
        images /= int(np.std(images))
        return images

    def pass_training_data(self):
        """The training data is standardized
        and passed to the main network.
        """
        images = self.normalize_images(training_images)

        # images = images[0:10]
        # labels = training_labels[0:10]

        images = images.reshape(images.shape[0], images.shape[1]**2)
        labels = training_labels.reshape(training_labels.shape[0], 1)
        # labels = labels.reshape(labels.shape[0], 1)
        data = np.hstack((images, labels))

        return data

    def pass_validation_data(self):
        """The validation data is standardized
        and passed to the main network.
        """
        return self.normalize_images(validation_images), validation_labels

    def pass_test_data(self):
        """Preprocess the test_data and
        pass it.
        """
        return self.normalize_images(test_images), test_labels

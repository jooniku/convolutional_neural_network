from src.mnist_data_processor import training_images, training_labels, validation_images, validation_labels, test_images, test_labels
import numpy as np


class InputLayer:
    """This class takes in data in correct, preprocessed format.

    Args:
        NeuralNetwork (_type_): _description_
    """

    def __init__(self):
        self.training_images = self._perform_data_augmentation(training_images)
        # self.mean = np.mean(self.training_images)
        # self.sd = np.std(self.training_images)

    def _perform_data_augmentation(self, images):
        """Perform data augmentation 
         for better generalization.
        """
        for i in range(len(images)):
            images[i] = self._random_rotation(images[i])

        return images

    def _random_rotation(self, image):
        """Rotate image within angle.
        """
        angle = np.random.uniform(-30, 30)
        return np.rot90(image, k=int(angle/90))

    def preprocess_data(self, images):
        return images
        images = images - self.mean
        images = images / self.sd

    def pass_training_data(self):
        """The training data is standardized
        and passed to the main network.
        """
        # print(self.training_images[0])
        images = self.preprocess_data(self.training_images)

        images = images[0:10000]
        labels = training_labels[0:10000]

        images = images.reshape(images.shape[0], images.shape[1]**2)
        # labels = training_labels.reshape(training_labels.shape[0], 1)
        labels = labels.reshape(labels.shape[0], 1)
        data = np.hstack((images, labels))

        return data

    def pass_validation_data(self):
        """The validation data is standardized
        and passed to the main network.
        """
        images = self.preprocess_data(validation_images)
        labels = validation_labels

        return images, labels

    def pass_test_data(self):
        """Preprocess the test_data and
        pass it.
        """
        images = self.preprocess_data(test_images)
        labels = test_labels

        return images, labels

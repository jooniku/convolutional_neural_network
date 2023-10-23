from src.mnist_data_processor import training_images, training_labels
import numpy as np

class InputLayer:
    """This class takes in data in correct, preprocessed format.

    Args:
        NeuralNetwork (_type_): _description_
    """

    def __init__(self) -> None:
        pass

    def _pass_training_data(self):
        images = training_images - np.mean(training_images)
        images = images / np.std(images)
        
        images = images[0:10]
        labels = training_labels[0:10]

        images = images.reshape(images.shape[0], images.shape[1]**2)
        labels = labels.reshape(labels.shape[0], 1)

        return images, labels
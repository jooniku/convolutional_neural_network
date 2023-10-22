from src.mnist_data_processor import training_images, training_labels


class InputLayer:
    """This class takes in data in correct, preprocessed format.

    Args:
        NeuralNetwork (_type_): _description_
    """

    def __init__(self) -> None:
        pass

    def _pass_training_data(self):
        images = training_images.reshape(training_images.shape[0], training_images.shape[1]**2)
        labels = training_labels.reshape(training_labels.shape[0], 1)

        return images, labels
from src.mnist_data_processor import training_images, training_labels, test_images, test_labels

class InputLayer:
    """This class takes in data in correct, preprocessed format.

    Args:
        NeuralNetwork (_type_): _description_
    """

    def __init__(self) -> None:
        pass


    def _pass_training_data(self):
        """Feeds the preprocessed data forward to the network.

        Returns:
            _type_: returns the data
        """
        return training_images, training_labels
    
#print(training_images[2])
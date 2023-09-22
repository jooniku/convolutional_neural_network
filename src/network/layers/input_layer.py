from network.mnist_data_processor import training_images, training_labels

class InputLayer:
    """This class takes in data in correct, preprocessed format.

    Args:
        NeuralNetwork (_type_): _description_
    """

    def __init__(self) -> None:
        pass


    def _pass_data(self):
        return training_images, training_labels
    
print(type(training_labels))
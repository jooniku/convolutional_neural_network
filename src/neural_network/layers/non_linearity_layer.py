from neural_network.neural_network import NeuralNetwork


class NonLinearLayer(NeuralNetwork):
    """This class represents the function to increase non-linearity in the neural network.
    It is implemented as a class from which one can choose different non-linearity functions.

    Args:
        NeuralNetwork (_type_): main neural network

    """
    def __init__(self) -> None:
        pass

    def relu(value: int):
        """Rectified Linear Unit (ReLU).
        Chooses the maximum value from (0, value).

        Args:
            value (int): value of position after convolution
        """
        return max(0, value)
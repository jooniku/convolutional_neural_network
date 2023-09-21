class NonLinearity:
    """This class has different functions to increase non-linearity in the neural network.
    The function is applied typically after a convolution.
    """

    def __init__(self) -> None:
        pass

    def _relu(self, value: float):
        """Rectified linear unit (ReLU) function to increase
        non-linearity.

        Args:
            value (int): value of position

        Returns:
            int : value, if it's greater than 0, else 0
        """
        return max(0, value)



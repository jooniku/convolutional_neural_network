import numpy as np

class NonLinearity:
    """This class has different functions to increase non-linearity in the neural network.
    The function is applied typically after a convolution.
    """

    def __init__(self) -> None:
        pass

    def _relu(self, image: np.array):
        """Rectified linear unit (ReLU) function to increase
        non-linearity.

        Args:
            value (int): value of position

        Returns:
            float : value, if it's greater than 0, else 0
        """
        for row in range(len(image)):
            for col in range(len(image[0])):
                image[row][col] = max(0.0, image[row][col])

        return image



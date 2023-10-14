import numpy as np


class NonLinearity:
    """This class has different functions to increase non-linearity in the neural network.
    The function is applied typically after a convolution.
    """

    def __init__(self) -> None:
        pass

    def _relu(self, images):
        """Rectified linear unit (ReLU) function to increase
        non-linearity.

        Args:
            value (int): value of position

        Returns:
            float : value, if it's greater than 0, else 0
        """
        height, width = images[0].shape
        for filtered_img in range(len(images)):
            for row in range(height):
                for col in range(width):
                    images[filtered_img][row][col] = max(0.0, images[filtered_img][row][col])

        return images

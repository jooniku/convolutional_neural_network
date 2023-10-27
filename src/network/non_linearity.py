import numpy as np


class NonLinearity:
    """This class has different functions to increase non-linearity in the neural network.
    The function is applied typically after a convolution.
    """

    def __init__(self):
        pass

    def _relu(self, images):
        """Leaky rectified linear unit (ReLU) function to increase
        non-linearity. 
        """
        height, width = images[0].shape
        for filtered_img in range(len(images)):
            for row in range(height):
                for col in range(width):
                    if images[filtered_img][row][col] < 0:
                        images[filtered_img][row][col] *= 0.00001
        return images

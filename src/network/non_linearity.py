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
        #print(image.shape)
        for row in range(len(image)):
            #print(image[row])
            for col in range(len(image)):
                image[row][col] = max(0.0, image[row][col])
        #print()
        #print()
        
        return np.array(image)



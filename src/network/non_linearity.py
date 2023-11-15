import numpy as np


class NonLinearity:
    """This class has different functions to increase non-linearity in the neural network.
    The function is applied typically after a convolution.
    """

    def __init__(self):
        self.received_inputs = {"conv_layer": [],
                                "fc_layer": []}

    def forward(self, image, layer_name):
        """Leaky rectified linear unit (ReLU) function to increase
        non-linearity. 
        """
        self.received_inputs[layer_name].append(image)
        image[image <= 0] *= 0.001
        return image

    def backpropagation(self, gradient, layer_name, layer_pos):
        """Backpropagation through ReLU function. If the 
        image during forward pass was less than 0, the
        gradient becomes a small constant. Layer position
        is referring to the layer after which the ReLU is done.
        """
        gradient[self.received_inputs[layer_name][layer_pos] <= 0] *= 0.001

        return gradient

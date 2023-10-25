from src.network.layers.classifier import Classifier
import numpy as np


class FullyConnectedLayer:
    """Class for the fully connected
    layer. This class handles
    the classifier also.
    """

    def __init__(self, num_of_classes, input_image_shape) -> None:
        self.number_of_classes = num_of_classes
        self.weight_matrix = None
        self.input_image_shape = input_image_shape

        self.classifier_function = Classifier()

        self.initialize_weight_matrix()

    def process(self, images):
        """This is a function to process the input image
        through the layer.
        """
        flattened_images = images.flatten()
        self.received_inputs = flattened_images
        activation = np.dot(flattened_images, self.weight_matrix) + self.bias

        return activation

    def initialize_weight_matrix(self):
        """Initialize weight matrix and
        bias for the layer.
        """
        size = self.input_image_shape[0]*self.input_image_shape[1]**2
        #size = 8 * 12 * 12
        self.weight_matrix = 0.01 * \
            np.random.randn(size, self.number_of_classes) \
            * np.sqrt(2.0 / size)
        self.bias = np.zeros((self.number_of_classes))

    def initialize_gradients(self):
        """Initialize the gradients for
        backpropagation.
        """
        self.gradient_weight = np.zeros_like(self.weight_matrix)
        self.bias_gradient = np.zeros_like(self.bias)

    def update_parameters(self, batch_size, learning_rate):
        """Update the parameters of the layer with
        stored gradients accumulated within batches.
        """
        self.gradient_weight /= batch_size
        self.bias_gradient /= batch_size

        self.weight_matrix -= learning_rate*self.gradient_weight
        self.bias -= learning_rate*self.bias_gradient

    def backpropagation(self, gradient_score, reg_strength):
        """Updates the weights in the weight matrix
        with the given gradients. 

        Args:
            gradient_score: Gradients from the previous layer
        """
        self.received_inputs = np.array(self.received_inputs)
        self.received_inputs = self.received_inputs.reshape(1, -1)
        gradient_score = gradient_score.reshape(1, -1)

        self.gradient_weight += np.dot(self.received_inputs.T, gradient_score)

        # L2 regularization
        self.gradient_weight += self.weight_matrix*reg_strength

        self.bias_gradient += np.sum(gradient_score)

        gradient_for_next_layer = np.dot(gradient_score, self.weight_matrix.T)

        return gradient_for_next_layer.reshape(self.input_image_shape)

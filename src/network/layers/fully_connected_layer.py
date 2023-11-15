from src.network.layers.classifier import Classifier
import numpy as np


class FullyConnectedLayer:
    """Class for the fully connected
    layer. This class handles
    the classifier also.
    """

    def __init__(self, num_of_classes, input_image_shape, non_linearity):
        self.number_of_classes = num_of_classes
        self.weight_matrixes = []
        self.biases = []
        self.received_inputs = []
        self.input_image_shape = input_image_shape
        self.num_dense_layers = 2

        self.classifier_function = Classifier()
        self.non_linearity = non_linearity

        self.initialize_weight_matrix()

    def process(self, images):
        """This is a function to process the input image
        through the layer.
        """
        flattened_image = images.flatten()

        for i in range(len(self.weight_matrixes)):
            self.received_inputs.append(flattened_image)
            flattened_image = np.dot(
                flattened_image, self.weight_matrixes[i]) + self.biases[i]
            if i < len(self.weight_matrixes)-1:
                flattened_image = self.non_linearity.forward(
                    flattened_image, "fc_layer")
        return flattened_image

    def initialize_weight_matrix(self):
        """Initialize weight matrix and
        bias for the layer.
        """
        size = self.input_image_shape[0]*self.input_image_shape[1]**2
        matrix, bias = self._create_weights(size, size)
        self.weight_matrixes.append(matrix)
        self.biases.append(bias)
        matrix, bias = self._create_weights(size, self.number_of_classes)
        self.weight_matrixes.append(matrix)
        self.biases.append(bias)

    def _create_weights(self, rows, columns):
        """Create a weight matrix and a 
        bias vector.
        """
        matrix = 0.1 * \
            np.random.randn(rows, columns) \
            * np.sqrt(2.0 / rows)
        bias = np.zeros((columns))

        return matrix, bias

    def initialize_gradients(self):
        """Initialize the gradients for
        backpropagation.
        """
        self.received_inputs = []
        self.gradient_weights = [np.zeros_like(self.weight_matrixes[i])
                                 for i in range(len(self.weight_matrixes))]
        self.bias_gradients = [np.zeros_like(self.biases[i])
                               for i in range(len(self.biases))]
        self.weight_mean_grads = [np.zeros_like(self.weight_matrixes[i])
                                  for i in range(len(self.weight_matrixes))]
        self.weight_grad_variances = [np.zeros_like(self.weight_matrixes[i])
                                      for i in range(len(self.weight_matrixes))]
        self.bias_mean_grads = [np.zeros_like(self.biases[i])
                                for i in range(len(self.biases))]
        self.bias_grad_variances = [np.zeros_like(self.biases[i])
                                    for i in range(len(self.biases))]

    def update_parameters(self, batch_size, learning_rate, beta1, beta2, clip_threshold, iterations):
        """Update the parameters of the layer with
        stored gradients accumulated within batches.
        """
        for dense in range(self.num_dense_layers):

            self.gradient_weights[dense] = self._clip_gradient(
                self.gradient_weights[dense], clip_threshold)
            self.bias_gradients[dense] = self._clip_gradient(
                self.bias_gradients[dense], clip_threshold)

            # Update moment vectors
            self.weight_mean_grads[dense] = beta1*self.weight_mean_grads[dense] \
                + (1-beta1)*(self.gradient_weights[dense]/batch_size)
            self.weight_grad_variances[dense] = beta2*self.weight_grad_variances[dense] \
                + (1-beta2) * (self.gradient_weights[dense]/batch_size)**2
            # Take the bias-corrected variables
            weight_mhat = self.weight_mean_grads[dense] / \
                (1 - beta1**(iterations+1))
            weight_vhat = self.weight_grad_variances[dense] / (
                1 - beta2**(iterations+1))
            # Update variable
            self.weight_matrixes[dense] -= learning_rate * \
                weight_mhat / (np.sqrt(weight_vhat)+1e-9)

            # Same for bias
            self.bias_mean_grads[dense] = beta1*self.bias_mean_grads[dense] \
                + (1-beta1)*(self.bias_gradients[dense]/batch_size)
            self.bias_grad_variances[dense] = beta2*self.bias_grad_variances[dense] \
                + (1-beta2) * (self.bias_gradients[dense]/batch_size)**2
            bias_mhat = self.bias_mean_grads[dense] / \
                (1 - beta1**(iterations+1))
            bias_vhat = self.bias_grad_variances[dense] / \
                (1 - beta2**(iterations+1))
            self.biases[dense] -= learning_rate * \
                bias_mhat / (np.sqrt(bias_vhat)+1e-9)

    def _clip_gradient(self, gradient, clip_threshold):
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > clip_threshold:
            scaling_factor = clip_threshold / gradient_norm
            gradient *= scaling_factor
        return gradient

    def backpropagation(self, gradient_score, reg_strength):
        """Updates the weights in the weight matrix
        with the given gradients. 

        Args:
            gradient_score: Gradients from the previous layer
        """
        self.received_inputs[-1] = np.array(
            self.received_inputs[-1]).reshape(1, -1)
        gradient_score = gradient_score.reshape(1, -1)

        self.gradient_weights[-1] += np.dot(
            self.received_inputs[-1].T, gradient_score)
        self.bias_gradients[-1] += np.sum(gradient_score)

        gradient_for_next_layer = np.dot(gradient_score,
                                         self.weight_matrixes[-1].T)

        gradient_for_next_layer = gradient_for_next_layer.reshape(-1, 1)
        gradient_for_next_layer = self.non_linearity.backpropagation(gradient_for_next_layer,
                                                                     "fc_layer", 0)

        self.received_inputs[-2] = np.array(
            self.received_inputs[-2]).reshape(1, -1)
        gradient_score = gradient_for_next_layer.reshape(1, -1)

        self.gradient_weights[-2] += np.dot(
            self.received_inputs[-2].T, gradient_score)
        self.bias_gradients[-2] += np.sum(gradient_score)

        gradient_for_next_layer = np.dot(gradient_score,
                                         self.weight_matrixes[-2].T)

        """
        gradient_for_next_layer = gradient_score
        for dense in range(len(self.weight_matrixes)-1, -1, -1):
            #print(gradient_for_next_layer)
            self.received_inputs[dense] = np.array(self.received_inputs[dense]).reshape(1, -1)
            gradient_for_next_layer = gradient_for_next_layer.reshape(1, -1)

            self.gradient_weights[dense] += np.dot(self.received_inputs[dense].T, gradient_for_next_layer)

            # L2 regularization
            self.gradient_weights[dense] += self.weight_matrixes[dense]*reg_strength

            self.bias_gradients[dense] += np.sum(gradient_for_next_layer)

            gradient_for_next_layer = np.dot(gradient_for_next_layer, 
                                            self.weight_matrixes[dense].T)
        """
        gradient_for_next_layer = gradient_for_next_layer.reshape(
            self.input_image_shape)

        return gradient_for_next_layer

from src.network.layers.classifier import Classifier
import numpy as np


class FullyConnectedLayer:
    """Fully connected layer.
    The final layer(s) in the network.
    """

    def __init__(self, num_of_classes, learning_step_size, reg_strength) -> None:
        self.number_of_classes = num_of_classes
        self.step_size = learning_step_size
        self.reg_strength = reg_strength
        self.received_inputs = []
        self.weight_matrix = None

        self.classifier_function = Classifier(
            learning_step_size=learning_step_size, reg_strength=reg_strength)

    def _process(self, images):
        """This is a function to process the input image
        through the layer.

        Args:
            image (_type_): input image

        Returns:
            _type_: dot product of input image and weights
        """

        # initialize weight matrix here to get correct dimensions
        self.__initialize_weigth_matrix(images.shape)
        flattened_images = images.flatten()
        self.received_inputs.append(flattened_images)
        activation = np.dot(flattened_images, self.weight_matrix) + self.bias
        
        result = self.classifier_function._compute_softmax_probabilities(
            activation)

        return result

    def __initialize_weigth_matrix(self, images_shape):
        """Initialize weight matrix for the FC layer.
        Is called when image is received so it is
        created with the correct dimensions.

        Args:
            image_shape (_type_): input image shape
        """
        if self.weight_matrix is None:
            self.input_image_shape = images_shape
            size = images_shape[0]*images_shape[1]**2
            self.weight_matrix = 0.01 * \
                        np.random.randn(size, self.number_of_classes) \
                        * np.sqrt(2.0 / size)
            self.bias = np.zeros((self.number_of_classes))

    def _update_parameters(self, gradient_score):
        """Updates the weights in the weight matrix
        with the given gradients. 

        Args:
            gradient_score (_type_): Gradient score from the previous layer

        Returns:
            _type_: Gradients for the next layer
        """
        self.received_inputs = np.array(self.received_inputs)
        gradient_weight = np.dot(self.received_inputs.T, gradient_score)

        # L2 regularization
        gradient_weight += self.weight_matrix*self.reg_strength
        self.weight_matrix += -self.step_size * gradient_weight

        self.bias += -np.sum(gradient_score, axis=0)

        gradient_for_next_layer = np.dot(gradient_score, self.weight_matrix.T)
        
        # take the mean gradient of the batch
        gradient_for_next_layer = np.mean(gradient_for_next_layer, axis=0)
        
        # empty received_inputs for next epoch
        self.received_inputs = []
        return gradient_for_next_layer.reshape(self.input_image_shape)

    def _compute_loss(self, probabilities, labels):
        """Calls the classifier function to compute
        loss, which is given (the function)
        as a parameter in the initialization.

        Args:
            images (_type_): _description_
            labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.classifier_function._compute_cross_entropy_loss(probabilities=probabilities, labels=labels)

    def _compute_gradient(self, probabilities, labels):
        """Calls classifier function to compute average
        gradient of a given batch. Classifier function is 
        determined in the initialization function.

        Args:
            images (_type_): _description_
            labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.classifier_function._compute_gradients(probabilities=probabilities, labels=labels)

from src.network.layers.softmax import SoftMaxClassifier
import numpy as np

class FullyConnectedLayer:
    """Fully connected layer.
    The final layer(s) in the network.
    """

    def __init__(self, num_of_classes, learning_step_size, reg_strength) -> None:
        self.number_of_classes = num_of_classes
        self.step_size = learning_step_size

        self.classifier_function = SoftMaxClassifier(learning_step_size=learning_step_size, reg_strength=reg_strength)


    def _process(self, image):
        """This is a function to process the input image
        through the layer.

        Args:
            image (_type_): input image

        Returns:
            _type_: dot product of input image and weights
        """

        # initialize weight matrix here to get correct dimensions
        self.__initialize_weigth_matrix(image.shape)
        
        flattened_image = image.flatten()
        self.received_input = flattened_image

        self.activations = np.dot(self.weight_matrix, flattened_image)

        return self.activations

    def __initialize_weigth_matrix(self, image_shape):
        """Initialize weight matrix for the FC layer.
        Is called when image is received so it is
        created with the correct dimensions.

        Args:
            image_shape (_type_): input image shape
        """
        self.input_image_shape = image_shape
        size = image_shape[0]**2
        self.weight_matrix = 0.01 *np.random.randn(self.number_of_classes, size) * np.sqrt(2.0 / 196)


    def _compute_loss(self, image, label):
        """_summary_

        Args:
            image (_type_): _description_
            kernel (_type_): _description_
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.classifier_function._cross_entropy_loss(image, label)

    def _compute_gradient(self, image, label):
        """_summary_

        Args:
            image (_type_): _description_
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.classifier_function._compute_gradient(image, label)
    
    def _update_parameters(self, gradient_score):
        """Updates the weights in the weight matrix
        with the given gradients. 

        Args:
            gradient_score (_type_): Gradient score from the previous layer

        Returns:
            _type_: Gradients for the next layer
        """
        gradient_score = gradient_score.reshape(1, -1)
        self.received_input = self.received_input.reshape(-1, 1)

        
        gradient_weight = np.dot(self.received_input, gradient_score)

        self.weight_matrix += -self.step_size * gradient_weight.T

        gradient_input = np.dot(gradient_score, self.weight_matrix)

        gradient_input = gradient_input.reshape(self.input_image_shape)

        return gradient_input
        

    def _compute_average_loss(self, images, labels):
        """Calls the classifier function to compute
        average loss, which is given (the function)
        as a parameter in the initialization.

        Args:
            images (_type_): _description_
            labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.classifier_function._compute_average_loss(images=images, labels=labels)

    def _compute_average_gradient(self, images, labels):
        """Calls classifier function to compute average
        gradient of a given batch. Classifier function is 
        determined in the initialization function.

        Args:
            images (_type_): _description_
            labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.classifier_function._compute_average_gradient(images=images, labels=labels)
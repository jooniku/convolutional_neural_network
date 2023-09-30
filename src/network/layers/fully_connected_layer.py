from layers.softmax import SoftMaxClassifier
import numpy as np

class FullyConnectedLayer:
    """_summary_

    Args:
        NeuralNetwork (_type_): _description_
    """

    def __init__(self, num_of_classes, learning_step_size, reg_strength) -> None:
        self.number_of_classes = num_of_classes
        self.step_size = learning_step_size
        # 196 is the size of the flattened array, this is to be changed to a variable
        self.weight_matrix = 0.01 *np.random.randn(self.number_of_classes, 196) * np.sqrt(2.0 / 196)
        self.classifier_function = SoftMaxClassifier(learning_step_size=learning_step_size, reg_strength=reg_strength)


    def _process(self, image):
        """_summary_

        Args:
            image (_type_): _description_

        Returns:
            _type_: _description_
        """
        flattened_image = image.flatten()

        return np.dot(self.weight_matrix, flattened_image)

    def _compute_loss(self, image, kernel, label):
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
        return self.classifier_function._compute_gradient(image, label=label)
    
    def _update_parameters(self, gradient_score):
        """_summary_

        Args:
            gradient_score (_type_): _description_

        Returns:
            _type_: _description_
        """
        gradient_weight = np.dot(self.weight_matrix.T, gradient_score)

        self.weight_matrix += -self.step_size*gradient_weight
        return gradient_weight


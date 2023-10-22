import numpy as np


class Classifier:

    def __init__(self, learning_step_size, reg_strength) -> None:
        self.learning_step_size = learning_step_size
        self.regularization_strength = reg_strength

    def _compute_softmax_probabilities(self, image):
        """This function computes the probabilities
        for each class using the softmax function.
        Works as the final layer.

        Args:
            image : Result array from FC layer

        Returns:
            probabilities : Probabilities of classes
        """
        # for numerical stability
        image -= np.max(image)

        exp_scores = np.exp(image)
        probabilities = (
            exp_scores / np.sum(exp_scores, axis=0, keepdims=True))

        return probabilities

    def _compute_cross_entropy_loss(self, probabilities, label):
        """This function computes the cross-entropy loss
        for the batch of images. 

        Args:
            probabilities : List of classification predictions
            labels : Correct labels

        Returns:
            Loss : Total loss over the complete batch
        """
        loss = -np.log(probabilities[label])
        
        return loss

    def _compute_gradients(self, probabilities, label):
        """This function computes the gradients
        of the batch. 

        Args:
            probabilities (_type_): _description_
            labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        probabilities[label] -= 1

        gradients = np.array(probabilities)

        return gradients

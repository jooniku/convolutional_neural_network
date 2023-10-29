import numpy as np


class Classifier:

    def __init__(self):
        pass

    def compute_probabilities(self, image):
        """This function computes the probabilities
        for each class using the softmax function.
        Works as the final layer.
        """
        # for numerical stability
        image -= np.max(image)

        exp_scores = np.exp(image)
        probabilities = (
            exp_scores / np.sum(exp_scores))

        return probabilities

    def compute_loss(self, probabilities, label):
        """This function computes the cross-entropy loss
        for the batch of images. 

        Args:
            probabilities : List of classification predictions
            labels : Correct labels

        Returns:
            Loss : Total loss over the complete batch
        """
        loss = -np.sum(np.log(probabilities[label]))

        return loss

    def compute_gradients(self, probabilities, label):
        """This function computes the gradients
        of the batch. 
        """
        probabilities[label] -= 1

        gradients = np.array(probabilities).reshape(1, 10)

        return gradients

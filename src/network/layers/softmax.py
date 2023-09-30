import numpy as np

class SoftMaxClassifier:

    def __init__(self, learning_step_size, reg_strength) -> None:
        self.learning_step_size = learning_step_size
        self.regularization_strength = reg_strength

    def _compute_probabilities(self, images):
        """_summary_

        Args:
            image (np.array): _description_

        Returns:
            _type_: _description_
        """
        probabilities = []
        for image in images:
            image -= np.max(image)
            exp_scores = np.exp(image)
            probabilities.append(list(exp_scores / np.sum(exp_scores, axis=0, keepdims=True)))

        return np.array(probabilities)

    def _create_one_hot_label(self, label):
        """_summary_

        Args:
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        labels = [0]*10
        for i in range(len(labels)):
            if i == label:
                labels[i] = 1
        return labels

    def _cross_entropy_loss(self, images, labels):
        """Calculates the cross-entropy loss of the network.

        Args:
            image (_type_): _description_
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        probabilities = self._compute_probabilities(images)

        prob_correct_class = -np.log(probabilities[range(len(images)),labels])

        average_data_loss = np.sum(prob_correct_class) / len(labels)

        return average_data_loss

    def _compute_gradient(self, images, labels):
        """

        Args:
            image (_type_): _description_
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        avr = self._cross_entropy_loss(images, labels)
        gradient = self._compute_probabilities(images)

        gradient[range(len(images)), labels] -= avr

        gradient /= len(images)

        return gradient


import numpy as np

class SoftMaxClassifier:

    def __init__(self, learning_step_size, reg_strength) -> None:
        self.learning_step_size = learning_step_size
        self.regularization_strength = reg_strength

    def _compute_probabilities(self, image:np.array):
        """_summary_

        Args:
            image (np.array): _description_

        Returns:
            _type_: _description_
        """
        image -= np.max(image)
        exp_scores = np.exp(image)

        probabilities = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
        return probabilities

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

    def _cross_entropy_loss(self, image, label):
        """Calculates the cross-entropy loss of the network.

        Args:
            image (_type_): _description_
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        labels = self._create_one_hot_label(label)

        probabilities = self._compute_probabilities(image)

        cross_entropy_loss = -np.sum(labels * np.log(probabilities))

        prob_correct_class = probabilities[np.argmax(labels)]

        if prob_correct_class > 0.5:
            print("Guessed correctly!")

        return cross_entropy_loss

    def _compute_gradient(self, image, label):
        """

        Args:
            image (_type_): _description_
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        probalilities = self._compute_probabilities(image)
        labels = self._create_one_hot_label(label)

        gradient = probalilities - labels
        print(gradient)
        return gradient

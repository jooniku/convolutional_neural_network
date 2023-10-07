import numpy as np

class SoftMaxClassifier:

    def __init__(self, learning_step_size, reg_strength) -> None:
        self.learning_step_size = learning_step_size
        self.regularization_strength = reg_strength

    def _compute_probabilities(self, image):
        """_summary_

        Args:
            image (np.array): _description_

        Returns:
            _type_: _description_
        """
        # for numerical stability
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
        label = [0]*10
        for i in range(len(label)):
            if i == label:
                label[i] = 1
        return label

    def _cross_entropy_loss(self, image, label):
        """Calculates the cross-entropy loss of the 
        prediction from the image.

        Args:
            image (_type_): _description_
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        probabilities = self._compute_probabilities(image)

        prob_correct_class = -np.log(probabilities[label])

        average_data_loss = np.sum(prob_correct_class) / 10

        return average_data_loss

    def _compute_gradient(self, image, label):
        """

        Args:
            image (_type_): _description_
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        gradient = self._compute_probabilities(image)

        gradient[label] -= 1

        gradient /= 10

        return gradient


    def _compute_average_loss(self, images, labels):
        """Calculates the average cross-entropy loss of the batch.

        Args:
            image (_type_): _description_
            label (_type_): _description_

        Returns:
            _type_: _description_
        """
        probabilities = self._compute_probabilities(images)

        prob_correct_class = -np.log(probabilities[range(len(images)), labels])

        average_data_loss = np.sum(prob_correct_class) / len(images)

        return average_data_loss

    def _compute_average_gradient(self, images, labels):
        probabilities = self._compute_probabilities(images)

        probabilities[range(len(images)), labels] -= 1


        average_gradient = np.mean(probabilities, axis=0)

        return average_gradient
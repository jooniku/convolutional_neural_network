import numpy as np

class SoftMaxClassifier:

    def __init__(self, learning_step_size, reg_strength) -> None:
        self.learning_step_size = learning_step_size
        self.regularization_strength = reg_strength

    def _compute_probabilities(self, image:np.array):
        exp_scores = np.exp(image)

        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probabilities

    def _compute_loss(self, image: np.array, kernel :np.array, label: int):
        probabilities = self._compute_probabilities(image=image)

        prob_correct_class = -np.log(probabilities[range(10), label])

        data_loss = np.sum(prob_correct_class) / 10

        regularization_loss = 0.5*self.regularization_strength*np.sum(kernel*kernel) + 0.5*self.regularization_strength*np.sum(image*image)


        return data_loss + regularization_loss

    def _compute_gradient(self, image, label):
        probabilities = self._compute_probabilities(image=image)

        probabilities[range(10), label] -= 1

        gradients = probabilities / 10

        return gradients


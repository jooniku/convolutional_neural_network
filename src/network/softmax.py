import numpy as np

class SoftMaxClassifier:

    def __init__(self) -> None:
        pass

    def _cross_entropy_loss(self, scores: np.array):
        exp_scores = np.exp(scores)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        prob_correct_class = -np.log(probabilities[range(),])

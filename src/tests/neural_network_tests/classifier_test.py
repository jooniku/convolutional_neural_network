import unittest
import numpy as np
from src.mnist_data_processor import training_images, training_labels
from src.network.layers.classifier import Classifier


class TestClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.classifier = Classifier(
            learning_step_size=0.01, reg_strength=0.001)

    def test_computing_probabilities_is_correct(self):
        image = [1, 2, 8]
        correct_probs = [0.001, 0.002, 0.997]
        probs = list(
            self.classifier._compute_softmax_probabilities(image=image))

        probs = [round(prob, 3) for prob in probs]

        self.assertEqual(probs, correct_probs)

    def test_cross_entropy_loss_is_correct(self):
        labels = [3, 2]
        images = [[1, 2, 8], [4, 9, 6]]

        probabilities = []
        for image in images:
            probabilities.append(
                self.classifier._compute_softmax_probabilities(image=image))

        correct_cross_loss = 0.05837

        cross_loss = self.classifier._compute_cross_entropy_loss(
            probabilities, labels)

        self.assertAlmostEqual(cross_loss, correct_cross_loss, 4)

    def test_gradients_are_correct(self):
        labels = [3, 2]
        images = [[1, 2, 8], [4, 9, 6]]

        probabilities = []
        for image in images:
            probabilities.append(
                self.classifier._compute_softmax_probabilities(image=image))

        grads = self.classifier._compute_gradients(
            probabilities, labels)
        correct_grad = [[0.0009088, 0.00247037, -0.0033792]
                        ,[0.00637746, -0.0535008, 0.0471234]]

        grads_sum = 0
        for row in grads:
            for col in row:
                grads_sum += col

        correct_sum = 0
        for row in correct_grad:
            for col in row:
                correct_sum += col

        self.assertAlmostEqual(grads_sum, correct_sum, 4)

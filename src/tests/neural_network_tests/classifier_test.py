import unittest
import numpy as np
from src.mnist_data_processor import training_images, training_labels
from src.network.layers.classifier import Classifier


class TestClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.classifier = Classifier()

    def test_computing_probabilities_is_correct(self):
        image = [1, 2, 8]
        correct_probs = [0.001, 0.002, 0.997]
        probs = list(
            self.classifier.compute_probabilities(image=image))

        probs = [round(prob, 3) for prob in probs]

        self.assertEqual(probs, correct_probs)

    def test_cross_entropy_loss_is_correct(self):
        labels = 2
        images = [1, 2, 8]

        probabilities = self.classifier.compute_probabilities(
            image=images)

        correct_cross_loss = 0.0033848989

        cross_loss = self.classifier.compute_loss(
            probabilities, labels)

        self.assertAlmostEqual(cross_loss, correct_cross_loss, 4)

    def test_gradients_are_correct(self):
        labels = 2
        images = [1, 2, 8]

        probabilities = self.classifier.compute_probabilities(
            image=images)

        grads = self.classifier.compute_gradients(
            probabilities, labels)
        correct_grad = [0.0009088, 0.00247037, -0.0033792]

        grads_sum = 0
        for row in grads:
            grads_sum += row

        correct_sum = 0
        for row in correct_grad:
            correct_sum += row

        self.assertAlmostEqual(grads_sum, correct_sum, 4)

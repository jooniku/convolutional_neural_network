import unittest
import numpy as np
from src.mnist_data_processor import training_images, training_labels
from src.network.layers.classifier import Classifier

class TestClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.classifier = Classifier(learning_step_size=0.01, reg_strength=0.001)

    
    def test_computing_probabilities_is_correct(self):
        image = [1, 2, 8]
        correct_probs = [0.001, 0.002, 0.997]
        probs = list(self.classifier._compute_softmax_probabilities(image=image))
        
        probs = [round(prob, 3) for prob in probs]

        self.assertEqual(probs, correct_probs)


    def test_cross_entropy_loss_is_correct(self):
        labels = [3, 2]
        images = [[1, 2, 8], [4, 9, 6]]

        probabilities = []
        for image in images:
            probabilities.append(self.classifier._compute_softmax_probabilities(image=image))

        correct_cross_loss = 0.05837

        cross_loss = self.classifier._compute_cross_entropy_loss(probabilities, labels)

        self.assertAlmostEqual(cross_loss, correct_cross_loss, 4)


    def test_average_gradient_is_correct(self):
        labels = [3, 2]
        images = [[1, 2, 8], [4, 9, 6]]

        probabilities = []
        for image in images:
            probabilities.append(self.classifier._compute_softmax_probabilities(image=image))

        avg_grad = self.classifier._compute_average_gradient(probabilities, labels)
        correct_grad = [0.003643, -0.0255, 0.02181]

        self.assertAlmostEqual(sum(avg_grad), sum(correct_grad), 4)
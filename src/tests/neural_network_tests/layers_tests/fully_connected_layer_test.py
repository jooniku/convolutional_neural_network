import unittest
import numpy as np
from src.mnist_data_processor import training_images, training_labels
from src.network.layers.fully_connected_layer import FullyConnectedLayer
from src.network.layers.classifier import Classifier


class TestFullyConnectedLayer(unittest.TestCase):

    def setUp(self) -> None:
        pass

"""
    def test_parameter_update_is_correct(self):
        self.classifier = Classifier()
        input_image_shape = (3, 3)
        self.layer = FullyConnectedLayer(
            num_of_classes=3, input_image_shape=input_image_shape)

        self.layer.weight_matrix = np.zeros((9, 3))
        self.layer.bias = np.zeros(3)
        label = 1

        input_image = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
        probs = self.layer.process(input_image)
        probs = self.classifier.compute_probabilities(probs)
        grads = self.classifier.compute_gradients(probs, label)

        self.layer.initialize_gradients()
        self.layer.backpropagation(grads, 0.001)

        self.layer.update_parameters(1, 0.01)

        updated_weight_matrix = self.layer.weight_matrix
        updated_bias = self.layer.bias
        correct_bias = np.sum(grads, keepdims=True)
        correct_update_weights = np.array([[0.00666667, -0.0033333, -0.0033333],
                                           [0.01333, -0.006666, -0.0066666],
                                           [0.020000, -0.00999999, -0.00999999],
                                           [0.033333, -0.0166666, -0.0166666],
                                           [0.0400000, -0.0199999, -0.0199999],
                                           [0.046666, -0.0233333, -0.023333],
                                           [0.05333, -0.0266666, -0.0266666],
                                           [0.060000, -0.02999999, -0.0299999]])

        self.assertAlmostEqual(sum(sum(updated_weight_matrix)), sum(
            sum(correct_update_weights)), 4)
        self.assertAlmostEqual(sum(updated_bias), sum(correct_bias), delta=3)
"""
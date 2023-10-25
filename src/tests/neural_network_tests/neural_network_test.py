import unittest
from src.network.neural_network import NeuralNetwork
from src.network.layers.pooling_layer import PoolingLayer
from src.network.layers.convolutional_layer import ConvolutionalLayer
from src.network.non_linearity import NonLinearity
from src.network.layers.input_layer import InputLayer
from src.mnist_data_processor import training_images, training_labels
import numpy as np


class TestNeuralNetwork(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_network_achieves_zero_loss(self):
        # test that network can achieve 0 loss

        nn = NeuralNetwork(batch_size=1,
                           epochs=10,
                           learning_rate=10,
                           reg_strength=0)

        images = training_images - np.mean(training_images)
        images = images / np.std(training_images)
        labels = training_labels[0]
        images = images[0]

        images = images.reshape(1, 28**2)
        labels = labels.reshape(1, 1)

        nn.training_data = np.hstack((images, labels))

        nn.train_network(save_network=False)
        latest_loss = nn.cost[-1]

        self.assertEqual(0, latest_loss)

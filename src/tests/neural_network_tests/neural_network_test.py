import unittest
from src.network.neural_network import NeuralNetwork
from src.network.layers.pooling_layer import PoolingLayer
from src.network.layers.convolutional_layer import ConvolutionalLayer
from src.network.non_linearity import NonLinearity
from src.mnist_data_processor import training_images, training_labels


class TestNeuralNetwork(unittest.TestCase):

    def setUp(self) -> None:
        self.kernel_size = 3
        self.num_of_convolution_layers = 1
        self.num_of_filters = 1
        self.training_image = training_labels[2]
        self.training_label = training_labels[2]

        self.neuralnetwork = NeuralNetwork()

import unittest
from src.network.neural_network import NeuralNetwork
from src.network.layers.pooling_layer import PoolingLayer
from src.network.layers.convolutional_layer import ConvolutionalLayer
from src.network.non_linearity import NonLinearity
from network.layers.mnist_data_processor import training_images, training_labels

class TestNeuralNetwork(unittest.TestCase):

    def setUp(self) -> None:
        self.kernel_size = 3
        self.num_of_convolution_layers = 2
        self.training_image = training_labels[2]
        self.training_label = training_labels[2]

        self.neuralnetwork = NeuralNetwork()

        self._initialize_custom_functions()

    def _initialize_custom_functions(self):
        """Here are customisable functions,
        as in one can use max pooling or average pooling.
        Change those here.
        """

        self.non_linearity_function = NonLinearity()._relu
        self.pooling_function = PoolingLayer(self.kernel_size)._max_pooling
        

    def test_training_label_is_correct(self):
        self.assertEqual(self.training_label, 4)


    def test_create_convolution_layers_creates_correct_object(self):
        self.num_of_convolution_layers = 1
        self.neuralnetwork._create_convolutional_layers()

        self.assertEqual(type(self.neuralnetwork.convolution_layers[0]), type(ConvolutionalLayer(kernel_size=self.kernel_size, stride_length=1)))

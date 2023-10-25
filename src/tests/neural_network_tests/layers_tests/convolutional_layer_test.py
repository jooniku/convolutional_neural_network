import unittest
import numpy as np
from src.network.layers.convolutional_layer import ConvolutionalLayer
from src.network.layers.fully_connected_layer import FullyConnectedLayer
from src.network.layers.classifier import Classifier
from src.mnist_data_processor import training_images, training_labels


class TestConvolutionalLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.training_image = training_images[2]
        self.training_label = training_labels[2]


    def test_convolution_works_correctly(self):
        # using hand-calculated convolution
        conv_layer = ConvolutionalLayer(
            num_of_filters=1, filter_size=3, stride_length=2)
        filter = np.array([[1, 1, 1],
                           [1, 0, 0],
                           [0, -1, -1]])
        bias = 1
        conv_layer.conv_in_shape = (1, 7, 7)
        test_img = np.array([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 2, 0, 0, 2, 0],
                             [0, 0, 1, 0, 1, 2, 0],
                             [0, 1, 0, 2, 0, 1, 0],
                             [0, 1, 2, 1, 0, 2, 0],
                             [0, 2, 1, 0, 1, 2, 0],
                             [0, 0, 0, 0, 0, 0, 0]])

        correct_result = np.array([[0, 2, -1],
                                   [-1, 2, 2],
                                   [4, 5, 4]])

        convoluted_img = conv_layer._convolute2d(test_img, filter, bias)

        self.assertEqual(np.sum(convoluted_img), np.sum(correct_result))


    def test_gradient_check_for_conv_layer(self):
        conv_layer = ConvolutionalLayer(
            num_of_filters=1, filter_size=3, stride_length=2)
        conv_layer.bias_vector = np.array([0, 0])

        #conv_layer.filters = np.array([[[1., 1., 1.],
        #                                [1., 0., 0.],
        #                                [0., -1., -1.]]])

        images = np.array([[[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 2, 0, 0, 2, 0],
                            [0, 0, 1, 0, 1, 2, 0],
                            [0, 1, 0, 2, 0, 1, 0],
                            [0, 1, 2, 1, 0, 2, 0],
                            [0, 2, 1, 0, 1, 2, 0],
                            [0, 0, 0, 0, 0, 0, 0]]])        

        fc = FullyConnectedLayer(10, (1, 13, 13))
        classifier = Classifier()

        neutral = conv_layer.add_2d_convolution(images)
        neutral = fc.process(neutral)
        neutral = classifier.compute_probabilities(neutral)
        neutral = classifier.compute_gradients(neutral, 1)

        numerical_gradients = np.zeros_like(conv_layer.filters)
        for i in range(len(conv_layer.filters[0])):
            for j in range(len(conv_layer.filters[0][i])):
                conv_layer.filters[0][i][j] -= 1e-7
                output_neg = conv_layer.add_2d_convolution(images)
                conv_layer.filters[0][i][j] += 2*1e-7
                output_pos = conv_layer.add_2d_convolution(images)
        
                neg = fc.process(output_neg)
                pos = fc.process(output_pos)

                neg_prob = classifier.compute_probabilities(neg)
                pos_prob = classifier.compute_probabilities(pos)

                numerical_gradients[0][i][j] = (classifier.compute_loss(pos_prob, 1) - classifier.compute_loss(neg_prob, 1)) / (2 * 1e-7)

        conv_layer.initialize_gradients()
        fc.initialize_gradients()

        conv_layer.backpropagation(fc.backpropagation(neutral, 0), 0)
        analytical_gradients = conv_layer.gradient_filters

        relative_error = abs(numerical_gradients - analytical_gradients) / (abs(numerical_gradients) + abs(analytical_gradients))[0]

        accumulated_error = np.sum(relative_error)
        print(accumulated_error)

        self.assertGreaterEqual(9*1e-6, accumulated_error)

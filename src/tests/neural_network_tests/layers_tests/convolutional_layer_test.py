import unittest
import numpy as np
from src.network.layers.convolutional_layer import ConvolutionalLayer
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
        test_img = np.array([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 2, 0, 0, 2, 0],
                             [0, 0, 1, 0, 1, 2, 0],
                             [0, 1, 0, 2, 0, 1, 0],
                             [0, 1, 2, 1, 0, 2, 0],
                             [0, 2, 1, 0, 1, 2, 0],
                             [0, 0, 0, 0, 0, 0, 0]])

        correct_result = np.array([[-1, 1, -2],
                                   [-2, 1, 1],
                                   [3, 4, 3]])

        convoluted_img = conv_layer._convolute2d(test_img, filter)

        self.assertEqual(np.sum(convoluted_img), np.sum(correct_result))


"""
    def test_adding_padding_adds_proper_amount(self):
        image1 = self.conv_layer._add_padding(self.training_image)
        image2 = self.conv_layer_large._add_padding(self.training_image)

        self.assertEqual((30, 30), image1.shape)
        self.assertEqual((32, 32), image2.shape)
"""

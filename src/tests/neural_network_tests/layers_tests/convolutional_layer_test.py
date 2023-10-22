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
    

    def test_backpropagation_is_correct(self):
        conv_layer = ConvolutionalLayer(
            num_of_filters=2, filter_size=3, stride_length=2)
        conv_layer.bias_vector = np.array([1, 2])
        conv_layer.filters = [np.array([[1, 1, 1],
                           [1, 0, 0],
                           [0, -1, -1]]), 
                           np.array([[2, 2, 2],
                           [2, 0, 0],
                           [0, -2, -2]])]

        
        images = np.array([[[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 2, 0, 0, 2, 0],
                             [0, 0, 1, 0, 1, 2, 0],
                             [0, 1, 0, 2, 0, 1, 0],
                             [0, 1, 2, 1, 0, 2, 0],
                             [0, 2, 1, 0, 1, 2, 0],
                             [0, 0, 0, 0, 0, 0, 0]],
                             [[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 2, 0, 0, 2, 0],
                             [0, 0, 1, 0, 1, 2, 0],
                             [0, 1, 0, 2, 0, 1, 0],
                             [0, 1, 2, 1, 0, 2, 0],
                             [0, 2, 1, 0, 1, 2, 0],
                             [0, 0, 0, 0, 0, 0, 0]]])
        
        

        output_dim = int((images.shape[1]-conv_layer.filter_size)/conv_layer.stride_length) + 1
        output_image = np.zeros((conv_layer.num_of_filters, output_dim, output_dim))
        conv_layer.conv_in_shape = images.shape
        for filter in range(len(conv_layer.filters)):
            filter_output = conv_layer._convolute2d(
                image=images[filter], filter=conv_layer.filters[filter], bias=conv_layer.bias_vector[filter])
            output_image[filter] += filter_output

                        
        gradients = np.array([[[1., 2., 3.],
                               [1., 2., 3.],
                               [1., 2., 3.]],
                               [[4., 5., 6.],
                               [4., 5., 6.],
                               [4., 5., 6.]]])
        
        dout = conv_layer._backpropagation(gradients, 1.0, 0.1)
        print(conv_layer.filters)

        self.assertEqual(1, 0)
    
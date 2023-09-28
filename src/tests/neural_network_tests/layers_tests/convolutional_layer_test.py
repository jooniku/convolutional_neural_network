import unittest
from src.network.layers.convolutional_layer import ConvolutionalLayer
from network.layers.mnist_data_processor import training_images, training_labels

class TestConvolutionalLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.conv_layer = ConvolutionalLayer()
        self.training_image = training_images[2]
        self.training_label = training_labels[2]

    def test_convolution_works_correctly(self):
        training_image = [[0.5]*28 for i in range(28)]
        bias_vector = [1, 1, 1]
        stride_length = 3
        weight_matrix = [[4, 2, 1], 
                         [5, 3, 2], 
                         [3, 5, 6]]
        img = self.conv_layer._add_2d_convolution(training_image, weight_matrix, bias_vector=bias_vector, stride_length=stride_length)
        print(img)

TestConvolutionalLayer().test_convolution_works_correctly()


"""
    def test_adding_padding_adds_proper_amount(self):
        image1 = self.conv_layer._add_padding(self.training_image)
        image2 = self.conv_layer_large._add_padding(self.training_image)

        self.assertEqual((30, 30), image1.shape)
        self.assertEqual((32, 32), image2.shape)
"""



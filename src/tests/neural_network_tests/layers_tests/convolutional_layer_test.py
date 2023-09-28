import unittest
from src.network.layers.convolutional_layer import ConvolutionalLayer
from src.data_processors.mnist_data_processor import training_images, training_labels

class TestConvolutionalLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.conv_layer = ConvolutionalLayer(kernel_size=3, stride_length=1)
        self.conv_layer_large = ConvolutionalLayer(kernel_size=6, stride_length=2)
        self.training_image = training_images[2]
        self.training_label = training_labels[2]

    def test_adding_padding_adds_proper_amount(self):
        image1 = self.conv_layer._add_padding(self.training_image)
        image2 = self.conv_layer_large._add_padding(self.training_image)

        self.assertEqual((30, 30), image1.shape)
        self.assertEqual((32, 32), image2.shape)

    
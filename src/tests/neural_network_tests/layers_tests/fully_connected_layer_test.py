import unittest
from src.mnist_data_processor import training_images, training_labels
from src.network.layers.fully_connected_layer import FullyConnectedLayer

class TestFullyConnectedLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.layer = FullyConnectedLayer(10, learning_step_size=0.01, reg_strength=0.001)

    


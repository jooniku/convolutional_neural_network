import numpy as np
import unittest
from src.network.non_linearity import NonLinearity


class TestNonLinearity(unittest.TestCase):

    def setUp(self) -> None:
        self.non_linearity = NonLinearity()

    def test_relu_returns_0_with_negative_input(self):
        test_image = np.array([[1, -32, 43, 5, 322, -1],
                               [0, 32, 43, 35, -20, -1],
                               [-3, 32, 43, 5, 322, -1],
                               [1, -32, 43, 5, 322, -1],
                               [0, 32, 43, 35, -20, -1],
                               [-3, 32, 43, 5, 322, -1]])

        correct_relu = np.array([[1, 0, 43, 5, 322, 0],
                                 [0, 32, 43, 35, 0, 0],
                                 [0, 32, 43, 5, 322, 0],
                                 [1, 0, 43, 5, 322, 0],
                                 [0, 32, 43, 35, 0, 0],
                                 [0, 32, 43, 5, 322, 0]])

        relu = self.non_linearity._relu(image=test_image)

        self.assertEqual(np.sum(relu), np.sum(correct_relu))

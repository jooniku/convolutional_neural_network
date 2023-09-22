import unittest
from src.network.non_linearity import NonLinearity


class TestNonLinearity(unittest.TestCase):

    def setUp(self) -> None:
        self.non_linearity = NonLinearity()

    def test_relu_returns_0_with_negative_input(self):
        value = -1
        relu = self.non_linearity._relu(value=value)

        self.assertEqual(relu, 0)

    def test_relu_returns_input_with_positive_input(self):
        value = 1
        relu = self.non_linearity._relu(value=value)

        self.assertEqual(relu, value)

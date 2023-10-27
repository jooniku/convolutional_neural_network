import unittest
import numpy as np
from src.network.layers.pooling_layer import PoolingLayer


class TestPoolingLayer(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_average_pooling_works_correctly(self):
        # hand pooled values, delta represents error threshold

        pool = PoolingLayer(kernel_size=2, stride=1)

        test_img = np.array([[[2, 4, 5, 3, 6, 8, 7],
                             [5, 2, 3, 4, 56, 6, 2],
                             [2, 4, 2, 3, 5, 4, 3],
                             [7, 6, 54, 3, 7, 8, 2],
                             [9, 9, 9, 7, 4, 3, 4],
                             [3, 1, 1, 1, 1, 1, 1],
                             [3, 4, 52, 5, 7, 5, 4]]])

        test_pool = pool.average_pooling(test_img)

        correct_pooling = np.array([[[13., 14., 15., 69., 76., 23.],
                                    [13., 11., 12., 68., 71., 15.],
                                    [19., 66., 62., 18., 24., 17.],
                                    [31., 78., 73., 21., 22., 17.],
                                    [22., 20., 18., 13., 9., 9.],
                                    [11., 58., 59., 14., 14., 11.]]])
        correct_pooling *= (1/4)

        self.assertAlmostEqual(np.sum(correct_pooling),
                               np.sum(test_pool), delta=0.05)

    def test_backprop_avg_pooling_is_correct(self):
        pool = PoolingLayer(kernel_size=2, stride=1)
        output_shape = 13

        grad = np.ones((1, 12, 12))

        out = pool.backpropagation_average_pooling(grad, output_shape)

        correct_out = np.ones((1, 13, 13))

        self.assertEqual(correct_out.shape, out.shape)

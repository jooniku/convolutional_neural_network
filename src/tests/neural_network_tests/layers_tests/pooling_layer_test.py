import unittest
import numpy as np
from src.network.layers.pooling_layer import PoolingLayer


class TestPoolingLayer(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_max_pooling_works_correctly(self):
        # hand pooled values, delta represents error threshold

        pool = PoolingLayer(kernel_size=2, stride=1)

        test_img = np.array([[[2, 4, 5, 3, 6, 8, 7],
                             [5, 2, 3, 4, 56, 6, 2],
                             [2, 4, 2, 3, 5, 4, 3],
                             [7, 6, 54, 3, 7, 8, 2],
                             [9, 9, 9, 7, 4, 3, 4],
                             [3, 1, 1, 1, 1, 1, 1],
                             [3, 4, 52, 5, 7, 5, 4]]])

        test_pool = pool.max_pooling(test_img)
        correct_pooling = np.array([[[5., 5., 5., 56., 56., 8.],
                                    [5., 4., 4., 56., 56., 6.],
                                    [7., 54., 54., 7., 8., 8.],
                                    [9., 54., 54., 7., 8., 8.],
                                    [9., 9., 9., 7., 4., 4.],
                                    [4., 52., 52., 7., 7., 5.]]])

        self.assertAlmostEqual(np.sum(correct_pooling),
                               np.sum(test_pool), delta=0.05)

    def test_backprop_max_pooling_is_correct(self):
        pool = PoolingLayer(kernel_size=2, stride=1)

        test_img = np.array([[[2, 4, 5, 3, 6, 8, 7],
                             [5, 2, 3, 4, 56, 6, 2],
                             [2, 4, 2, 3, 5, 4, 3],
                             [7, 6, 54, 3, 7, 8, 2],
                             [9, 9, 9, 7, 4, 3, 4],
                             [3, 1, 1, 1, 1, 1, 1],
                             [3, 4, 52, 5, 7, 5, 4]]])

        pooled_img = pool.max_pooling(test_img)

        grad = np.ones((1, 7, 7))

        out = pool.max_pooling_backpropagation(grad, 0)

        correct_out = np.array([[0., 1., 1., 0., 0., 1., 0.],
                                [1., 1., 1., 0., 1., 1., 0.],
                                [0., 1., 1., 1., 0., 0., 1.],
                                [0., 1., 1., 0., 1., 1., 0.],
                                [1., 1., 1., 1., 1., 0., 0.],
                                [0., 0., 0., 1., 0., 1., 1.],
                                [0., 1., 1., 0., 1., 0., 0.]])

        self.assertEqual(np.sum(correct_out), np.sum(out))

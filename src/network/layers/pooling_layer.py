import numpy as np


class PoolingLayer:
    """Pooling layer for the neural network. 
    Condences the result from convolution layer and non-linearity function. 
    """

    def __init__(self, kernel_size, stride):
        self.pooling_kernel_size = kernel_size
        self.stride_length = stride
        self.received_input = None

    def max_pooling(self, image):
        num_images, in_dim, _ = image.shape
        self.received_input = image

        out_dim = int((in_dim-self.pooling_kernel_size)/self.stride_length)+1

        output = np.zeros((num_images, out_dim, out_dim))

        for i in range(num_images):
            image_y = out_y = 0
            while image_y + self.pooling_kernel_size <= in_dim:
                image_x = out_x = 0
                while image_x + self.pooling_kernel_size <= in_dim:
                    output[i, out_y, out_x] = np.max(image[i, image_y:image_y+self.pooling_kernel_size,
                                                           image_x:image_x+self.pooling_kernel_size])
                    image_x += self.stride_length
                    out_x += 1
                image_y += self.stride_length
                out_y += 1
        return output

    def max_pooling_backpropagation(self, gradient, output_shape):
        out_dim = output_shape
        output_grad = np.zeros((gradient.shape[0], out_dim, out_dim))

        for i in range(len(gradient)):
            d_y = out_y = 0
            while d_y + self.pooling_kernel_size <= out_dim:
                d_x = out_x = 0
                while d_x + self.pooling_kernel_size <= out_dim:
                    window = self.received_input[i, d_y:d_y+self.pooling_kernel_size,
                                                                    d_x:d_x+self.pooling_kernel_size]
                    x, y = np.unravel_index(window.argmax(), window.shape)
                    output_grad[i, d_y+y, d_x+x] = gradient[i, out_y, out_x]
                    d_x += self.stride_length
                    out_x += 1
                d_y += self.stride_length
                out_y += 1
        return output_grad
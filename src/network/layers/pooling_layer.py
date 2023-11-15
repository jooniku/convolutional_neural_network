import numpy as np


class PoolingLayer:
    """Pooling layer for the neural network. 
    Condences the result from convolution layer and non-linearity function. 
    """

    def __init__(self, kernel_size, stride):
        self.pooling_kernel_size = kernel_size
        self.stride_length = stride
        self.received_inputs = []

    def max_pooling(self, image):
        num_images, in_dim, _ = image.shape
        self.received_inputs.append(image)

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

    def max_pooling_backpropagation(self, gradient, layer_pos):
        input_image = self.received_inputs[layer_pos]
        out_dim = input_image.shape[1]
        output_grad = np.zeros(input_image.shape)

        for i in range(len(gradient)):
            d_y = out_y = 0
            while d_y + self.pooling_kernel_size <= out_dim:
                d_x = out_x = 0
                while d_x + self.pooling_kernel_size <= out_dim:
                    window = input_image[i, d_y:d_y+self.pooling_kernel_size,
                                         d_x:d_x+self.pooling_kernel_size]
                    x, y = np.unravel_index(window.argmax(), window.shape)
                    output_grad[i, d_y+y, d_x+x] = gradient[i, out_y, out_x]
                    d_x += self.stride_length
                    out_x += 1
                d_y += self.stride_length
                out_y += 1
        if layer_pos == 0:
            self.received_inputs = []
        return output_grad

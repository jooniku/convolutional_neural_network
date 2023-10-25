import numpy as np


class PoolingLayer:
    """Pooling layer for the neural network. 
    Condences the result from convolution layer and non-linearity function. 
    """

    def __init__(self, kernel_size):
        self.pooling_kernel_size = kernel_size
        self.stride_length = 1

    def max_pooling(self, image: np.array):
        """Max pooling algorithm. Currently not tested with
        different kernel sizes or stride lengths. 
        """
        pooled_image = []
        kernel_y_pos = 0
        while kernel_y_pos <= (len(image) - self.pooling_kernel_size):
            pooled_img_sublist = []
            kernel_x_pos = 0
            while kernel_x_pos <= (len(image[0]) - self.pooling_kernel_size):
                maximum_value = -np.Infinity

                for row in range(self.pooling_kernel_size):
                    for column in range(self.pooling_kernel_size):
                        maximum_value = max(
                            maximum_value, image[kernel_y_pos+row][kernel_x_pos+column])

                kernel_x_pos += self.stride_length
                pooled_img_sublist.append(maximum_value)

            kernel_y_pos += self.stride_length
            pooled_image.append(pooled_img_sublist)

        layer_activation = np.array(pooled_image)

        return layer_activation

    def average_pooling(self, images):
        """Computes the average pooling for a given input.
        For each local area, which is determined by the
        pooling kernel, takes the average of values and
        represents the local area with that value.
        """
        self.input_shape = images[0].shape
        output_images = []
        for filtered_img in range(len(images)):
            pooled_image = []
            kernel_y_pos = 0
            while kernel_y_pos <= (len(images[filtered_img]) - self.pooling_kernel_size):
                pooled_img_sublist = []
                kernel_x_pos = 0
                while kernel_x_pos <= (len(images[filtered_img][0]) - self.pooling_kernel_size):
                    local_sum = 0
                    for row in range(self.pooling_kernel_size):
                        for column in range(self.pooling_kernel_size):
                            local_sum += images[filtered_img][kernel_y_pos +
                                                              row][kernel_x_pos+column]
                    kernel_x_pos += self.stride_length
                    pooled_img_sublist.append(
                        local_sum / self.pooling_kernel_size**2)

                kernel_y_pos += self.stride_length
                pooled_image.append(pooled_img_sublist)

            pooled_image = np.array(pooled_image)
            output_images.append(pooled_image)
        return np.array(output_images)

    def backpropagation_average_pooling(self, gradient_input, output_shape):
        """Backpropagation through the average pooling
        function. Essentially a de-pooling function.
        Gives the average pooling value for the whole 
        local are from which it is computed from.
        """
        output = []
        for filter_i in range(gradient_input.shape[0]):
            gradients = np.zeros((output_shape, output_shape))
            height, width = gradient_input.shape[1], gradient_input.shape[2]
            num_of_contributing_pos = self.pooling_kernel_size**2

            for row in range(height):
                for column in range(width):
                    gradient_value = gradient_input[filter_i][row][column] / \
                        num_of_contributing_pos
                    gradients[row:row+self.pooling_kernel_size,
                              column:column+self.pooling_kernel_size] += gradient_value

            output.append(gradients)

        return np.array(output)

import numpy as np

class PoolingLayer:
    """Pooling layer for the neural network. 
    Condences the result from convolution layer and non-linearity function. 
    """

    def __init__(self, kernel_size):
        self.pooling_kernel_size = kernel_size
        self.stride_length = 2

    def _max_pooling(self, image: np.array):
        """Max pooling algorithm. Currently not tested with
        different kernel sizes or stride lengths. 

        Args:
            image (np.array): image to pool

        Returns:
            _type_: A pooled image
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
                            maximum_value = max(maximum_value, image[kernel_y_pos+row][kernel_x_pos+column])

                kernel_x_pos += self.stride_length
                pooled_img_sublist.append(maximum_value)

            kernel_y_pos += self.stride_length
            pooled_image.append(pooled_img_sublist)

        layer_activation = np.array(pooled_image)

        return layer_activation

    def _average_pooling(self, image: np.array):
        """Computes the average pooling for a given input.
        For each local area, which is determined by the
        pooling kernel, takes the average of values and
        represents the local area with that value.

        Args:
            image (np.array): _description_

        Returns:
            _type_: _description_
        """
        pooled_image = []
        kernel_y_pos = 0
        while kernel_y_pos <= (len(image) - self.pooling_kernel_size):
            pooled_img_sublist = []
            kernel_x_pos = 0
            while kernel_x_pos <= (len(image[0]) - self.pooling_kernel_size):
                local_sum = 0
                
                for row in range(self.pooling_kernel_size):
                        for column in range(self.pooling_kernel_size):
                            local_sum += image[kernel_y_pos+row][kernel_x_pos+column]

                kernel_x_pos += self.stride_length
                pooled_img_sublist.append(local_sum / self.pooling_kernel_size**2)

            kernel_y_pos += self.stride_length
            pooled_image.append(pooled_img_sublist)

        layer_activation = np.array(pooled_image)

        return layer_activation


    def _backpropagation_average_pooling(self, gradient_input, output_shape):
        """Backpropagation through the average pooling
        function. Essentially a de-pooling function.
        Gives the average pooling value for the whole 
        local are from which it is computed from.

        Args:
            gradient_input (_type_): _description_
            output_shape (_type_): _description_

        Returns:
            _type_: _description_
        """
        output_height, output_width = gradient_input.shape
        gradients = np.zeros(output_shape)

        num_of_contributing_pos = self.pooling_kernel_size**2
        for row in range(output_height):
            for column in range(output_width):
                gradient_value = gradient_input[row, column] / num_of_contributing_pos

                for i in range(self.pooling_kernel_size):
                    for j in range(self.pooling_kernel_size):
                        gradients[row + i, column + j] += gradient_value
            
        return gradients


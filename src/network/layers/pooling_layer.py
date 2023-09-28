import numpy as np

class PoolingLayer:
    """Pooling layer for the neural network. 
    Condences the result from convolution layer and non-linearity function. 

    Args:
        NeuralNetwork : main neural network
    """

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
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
        while kernel_y_pos <= (len(image) - self.kernel_size):
            pooled_img_sublist = []
            kernel_x_pos = 0
            while kernel_x_pos <= (len(image[0]) - self.kernel_size):
                maximum_value = -np.Infinity
                
                for row in range(self.kernel_size):
                        for column in range(self.kernel_size):
                            num = image[row][column]
                            maximum_value = max(maximum_value, image[kernel_y_pos+row][kernel_x_pos+column])

                kernel_x_pos += self.stride_length
                pooled_img_sublist.append(maximum_value)

            kernel_y_pos += self.stride_length
            pooled_image.append(pooled_img_sublist)
        
        return np.array(pooled_image)

    def _average_pooling(image: np.array):
        pass


pl = PoolingLayer(2)

#training_image = np.array([[0.5]*10 for i in range(10)])
training_image = np.array([[1, 2, 3, 2, 5, 2, 6, 7, 4, 5],
                           [2, 4, 6, 7, 2, 4, 5, 3, 6, 2],
                           [4, 6, 8, 5, 2, 2, 5, 7, 8, 9],
                           [1, 10, 3, 7, 5, 2, 1, 7, 4, 5],
                           [2, 4, 6, 7, 2, 4, 5, 3, 6, 2],
                           [4, 6, 8, 5, 2, 2, 5, 7, 8, 9],
                           [1, 2, 3, 2, 12, 2, 22, 7, 4, 5],
                           [2, 4, 6, 7, 2, 4, 5, 3, 6, 2],
                           [99, 6, 8, 5, 2, 2, 5, 7, 8, 9],
                           [2, 4, 6, 7, 2, 4, 5, 3, 6, 2]])
print(pl._max_pooling(image=training_image))
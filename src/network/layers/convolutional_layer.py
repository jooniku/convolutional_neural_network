import numpy as np

class ConvolutionalLayer:

    def __init__(self, kernel_size) -> None:
        self.kernel_size = kernel_size

    def __create_kernel(self):
        """Creates the kernel that holds the weights.
        The weights are initialized to small random numbers.
        """
        # create weight matrix using numbers from numpys samples of "standard normal" distribution
        weight_matrix = 0.01 * np.random.randn(self.kernel_size, self.kernel_size) * np.sqrt(2.0 / self.kernel_size)
        bias_vector = np.array([0.01]*3)

        return weight_matrix, bias_vector


    def _add_padding(self, image: np.array):
        """Adds zero-padding for the image to make sure the
        convolutions work.

        Args:
            image (np.array): array representation of image

        Returns:
            _type_: padded image
        """
        needed_padding = (self.kernel_size*10 - image.shape[0]) // 2

        return np.pad(image, pad_width=needed_padding)
    

    def _add_2d_convolution(self, raw_image: np.array, kernel: np.array, bias_vector: np.array, stride_length: int):
        """For the image, multiply the image data with the
        weight matrix. Returns the multiplied image.

        Args:
            image (np.array): image to convolute
        """
        image = self._add_padding(raw_image)

        kernel_y_pos = 0
        while kernel_y_pos < (30 - stride_length-1):
            for row in range(self.kernel_size):
                kernel_x_pos = 0
                while kernel_x_pos < (30 - stride_length-1):
                    for column in range(self.kernel_size):
                        image[kernel_y_pos + row][kernel_x_pos + column] = image[kernel_y_pos + row][kernel_x_pos + column]*kernel[row][column] + bias_vector[column]
                    kernel_x_pos += stride_length
            kernel_y_pos += stride_length

        print(image.shape)
        return image
    

    
from matplotlib import pyplot as plt

from mnist_data_processor import training_images

train_img = training_images[7]
training_image = np.array([[0.5]*28 for i in range(28)])
bias_vector = np.array([0, 0, 0])
stride_length = 2
weight_matrix = np.array([[0, -1, 0], 
                        [-1, 5, -1], 
                        [0, -1, 0]])
conv = ConvolutionalLayer(3)
conv._add_2d_convolution(training_image, weight_matrix, bias_vector=bias_vector, stride_length=stride_length)

conv = ConvolutionalLayer(3)
data = conv._add_2d_convolution(train_img, kernel=weight_matrix, bias_vector=bias_vector, stride_length=stride_length)


plt.imshow(data, interpolation='nearest')
plt.show()
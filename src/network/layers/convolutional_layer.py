import numpy as np

class ConvolutionalLayer:

    def __init__(self, kernel_size, stride_length) -> None:
        self.kernel_size = kernel_size
        self.stride_length = stride_length

        self.__create_kernel()

    def __create_kernel(self):
        """Creates the kernel that holds the weights.
        The weights are initialized to small random numbers.
        """
        # create kernel using numbers from numpys samples of "standard normal" distribution
        self.kernel = 0.01 * np.random.randn(self.kernel_size, self.kernel_size) * np.sqrt(2.0 / self.kernel_size)
        self.bias_vector = np.array([0.01]*self.kernel_size)


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
    
    def _add_2d_convolution(self, raw_image: np.array):
        image = np.reshape(self._add_padding(raw_image), (9, 10**2))
        kernel = np.reshape(self.kernel, (1, 9))

        convoluted_img = np.dot(kernel, image)

        convoluted_img = np.reshape(convoluted_img, (10, 10))
        print(convoluted_img)
        return convoluted_img

    def _add_2d_convolution1(self, raw_image: np.array):
        """This is the main convolution function. Using
        this class' kernel, slide the kernel across the
        image and add the bias vector. At each position, 
        calculate the dot product with the kernel and
        image.

        Args:
            image (np.array): image to convolute
        """
        return self._calculate_sums(self._calculate_dot_product(raw_image=raw_image))


    def _calculate_dot_product(self, raw_image: np.array):
        image = self._add_padding(raw_image)

        kernel_y_pos = 0
        while kernel_y_pos < (30 - self.stride_length-1):
            for row in range(self.kernel_size):
                kernel_x_pos = 0
                while kernel_x_pos < (30 - self.stride_length-1):
                    for column in range(self.kernel_size):
                        image[kernel_y_pos + row][kernel_x_pos + column] = image[kernel_y_pos + row][kernel_x_pos + column]*self.kernel[row][column]
                    kernel_x_pos += self.stride_length
            kernel_y_pos += self.stride_length
        return image
    

    def _calculate_sums(self, image: np.array):
        output_image = []
        kernel_y_pos = 0
        while kernel_y_pos <= (len(image) - self.kernel_size):
            output_image_sublist = []
            kernel_x_pos = 0
            while kernel_x_pos <= (len(image[0]) - self.kernel_size):
                local_area_sum = 0
                for row in range(self.kernel_size):
                        for column in range(self.kernel_size):
                            local_area_sum += image[kernel_y_pos+row][kernel_x_pos+column]

                output_image_sublist.append(local_area_sum + self.bias_vector)
                kernel_x_pos += self.stride_length
            print(output_image_sublist)
            output_image.append(output_image_sublist)
            kernel_y_pos += self.stride_length

        return np.array(output_image)

"""    
from matplotlib import pyplot as plt

from mnist_data_processor import training_images

train_img = training_images[7]
training_image = np.array([[0.5]*28 for i in range(28)])
bias_vector = np.array([0, 0, 0])
stride_length = 2
kernel = np.array([[0, -1, 0], 
                        [-1, 5, -1], 
                        [0, -1, 0]])
conv = ConvolutionalLayer(3)
conv._add_2d_convolution(training_image, kernel, bias_vector=bias_vector, stride_length=stride_length)

conv = ConvolutionalLayer(3)
data = conv._add_2d_convolution(train_img, kernel=kernel, bias_vector=bias_vector, stride_length=stride_length)


plt.imshow(data, interpolation='nearest')
plt.show()

"""
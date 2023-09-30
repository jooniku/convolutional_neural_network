import numpy as np
import time


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
        self.kernel = 0.01 *np.random.randn(self.kernel_size, self.kernel_size) * np.sqrt(2.0 / self.kernel_size)
        #self.kernel = np.array([[-1, -1, 0],
        #                        [-1, -1, 1],
        #                        [1, -1, -1]])
        #self.kernel = np.array([[-1, -1, -1],
        #                       [-1, 8, -1],
        #                       [-1, -1, -1]])
        self.bias_vector = [0.01]*((28-self.stride_length + 2*2)//2+1)

        #self.bias_vector = 1



    def _add_2d_convolution(self, image: np.array):
        """This is the main convolution function. Using
        this class' kernel, slide the kernel across the
        image and add the bias vector. At each position, 
        calculate the dot product with the kernel and
        image.

        Args:
            image (np.array): image to convolute
        """
        self.received_input = image
        output_image = []

        kernel_y_pos = 0
        while kernel_y_pos <= (len(image) - self.kernel_size):
            kernel_x_pos = 0
            output_image_sublist = []
            while kernel_x_pos <= (len(image) - self.kernel_size):
                local_area_sum = 0
                for row in range(self.kernel_size):
                    for column in range(self.kernel_size):
                        local_area_sum += image[kernel_y_pos + row][kernel_x_pos + column]*self.kernel[row][column]
                output_image_sublist.append(local_area_sum)
                kernel_x_pos += self.stride_length
            output_image.append(output_image_sublist)
            kernel_y_pos += self.stride_length
        
        return np.array(output_image)
    

    def _update_parameters(self, loss_score, reg_strength: float, step_size: float):
        """Update the kernel and bias vector parameters in backpropagation.

        Args:
            gradient_scores (np.array): _description_
            reg_strength (float): _description_
            step_size (float): _description_
        """
        d_kernel = np.dot(self.kernel.T, loss_score)

        #d_kernel += self.kernel*reg_strength

        self.kernel += -step_size*d_kernel
        #self.bias_vector += -step_size*d_bias_vector
        #print(((self.bias_vector - ss) / ss)*100)




img = np.array([[0, 0, 0, 2, 1],
                [2, 0, 0, 2, 2],
                [0, 2, 2, 2, 2],
                [0, 2, 2, 0, 0],
                [0, 2, 2, 1, 0]])

#print(ConvolutionalLayer(kernel_size=3, stride_length=2)._add_2d_convolution(img))



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
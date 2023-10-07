import numpy as np
from scipy.linalg import toeplitz

class ConvolutionalLayer:

    def __init__(self, filter_size, stride_length, num_of_filters):
        self.filter_size = filter_size
        self.num_of_filters = num_of_filters
        self.stride_length = stride_length

        self.__create_filter()

    def __create_filter(self):
        """Creates the filter that holds the weights.
        The weights are initialized to small random numbers.
        """
        # create filter using numbers from numpys samples of "standard normal" distribution
        self.filters = [0.01 *np.random.randn(self.filter_size, self.filter_size) * np.sqrt(2.0 / self.filter_size)
                        for i in range(self.num_of_filters)]
        
        self.received_inputs = []

    def _add_padding(self, image: np.array):
        """Adds zero-padding for the image to make sure the
        operations work. All images are padded to be the
        same shape as the original input.

        Args:
            image (np.array): array representation of image

        Returns:
            _type_: padded image
        """
        needed_padding = (28 - len(image)) // 2
        
        return np.pad(image, pad_width=needed_padding)


    def _add_2d_convolution(self, image):
        """This function is called for convolution. It
        calls the actual convolution function with 
        all filters in the convolutional layer.

        Args:
            image (np.array): image to convolute
        """
        for filter in self.filters:
            image = self._convolute2d(image=self._add_padding(image), filter=filter)
        return image


    def _convolute2d(self, image: np.array, filter):
        """This is the main convolution function. Using
        this class' filter, slide the filter across the
        image and add the bias vector. At each position, 
        calculate the sum of the dot products between filter
        and the local area.

        Args:
            image (np.array): image to convolute
            filter (np.array): filter to convolute with
        """

        self.received_inputs.append(image)
        output_image = []

        filter_y_pos = 0
        while filter_y_pos <= (len(image) - len(filter)):
            filter_x_pos = 0
            output_image_sublist = [] # these act as the rows in the output image
            while filter_x_pos <= (len(image) - len(filter)):
                local_area_sum = 0
                for row in range(len(filter)):
                    for column in range(len(filter)):
                        local_area_sum += image[filter_y_pos + row][filter_x_pos + column]*filter[row][column]
                output_image_sublist.append(local_area_sum)
                filter_x_pos += self.stride_length
            output_image.append(output_image_sublist)
            filter_y_pos += self.stride_length
        
        output = np.array(output_image)

        return output

    def _backpropagation(self, gradient_input, step_size):
        """This function takes care of the backpropagation for the
        convolution layer. For each filter in the layer, it calls
        the _get_filter_gradient function to get the gradients for the filters
        and then updates the filters.

        Args:
            gradient_input (_type_): gradient input from the previous layer
            step_size (_type_): learning rate

        Returns:
            gradient_input: the last gradient input goes to the next layer
        """
        for filter_i in range(len(self.filters)):
            gradient_input, gradient_filter = self._get_filter_gradient(gradient_input=gradient_input,
                                                                             filter=self.filters[filter_i],
                                                                             received_input=self.received_inputs[filter_i])
            self.filters[filter_i] += -step_size*gradient_filter

        
        return gradient_input

    def _get_filter_gradient(self, gradient_input, filter, received_input):
        """Update the filter and bias vector parameters in backpropagation.

        Args:
            gradient_scores (np.array): _description_
            reg_strength (float): _description_
            step_size (float): _description_
        """
        gradient_filter = np.zeros_like(filter)
        for height in range(len(gradient_input)):
            for width in range(len(gradient_input)):
                # gets a local region of the filters size
                input_region = received_input[height:height+len(filter), width:width+len(filter)]
                
                if input_region.shape != filter.shape:
                    break

                region_gradient = gradient_input[height, width] * input_region
                
                gradient_filter += region_gradient

                gradient_filter /= len(gradient_input)**2


        return gradient_filter, gradient_filter


"""

cl  = ConvolutionalLayer(4, 2, 2)
cl.received_inputs.append(np.full((28, 28), 2))
image = np.ones((28, 28))

cl._backpropagation(image, 0.1)
"""
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

    def _convolute2d_matrix(self, image: np.array, filter: np.array):
        num_input_row, num_input_col = image.shape
        num_filter_row, num_filter_col = filter.shape

        num_output_row = num_input_row + num_filter_row - 1
        num_output_col = num_input_col + num_filter_col - 1

        # zero-pad filter so that the padding is on the top and right of values
        padded_filter = np.pad(filter, ((num_output_row - num_filter_row, 0),
                                        (0, num_output_col - num_filter_col)), 
                                        "constant", constant_values=0)
        


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
        filter_height, filter_width = filter.shape


        for height in range(len(gradient_input)-filter_height):
            for width in range(len(gradient_input)-filter_width):
                # gets a local region of the filters size
                input_region = received_input[height:height+filter_height, width:width+filter_width]
                
                region_gradient = gradient_input[height, width] * input_region
                
                gradient_filter += region_gradient

        gradient_filter /= len(gradient_input)**2


        return gradient_filter


    def _convolute2d_matrix(self, image: np.array, filter: np.array):
        
        doubly_blocked_filter, output_shape = self.__create_doubly_blocked_toeplitz_filter(image.shape, filter)

        flat_image = np.reshape(np.flipud(image), (image.shape[0]*image.shape[1]))
        
        result_vector = np.dot(doubly_blocked_filter, flat_image)
        

        result = np.flipud(result_vector.reshape(output_shape))

        #print(result)


    def __create_doubly_blocked_toeplitz_filter(self, input_shape, filter):
        num_input_row, num_input_col = input_shape
        num_filter_row, num_filter_col = filter.shape

        num_output_row = num_input_row + num_filter_row - 1
        num_output_col = num_input_col + num_filter_col - 1

        # zero-pad filter so that the padding is on the top and right of values
        padded_filter = np.pad(filter, ((num_output_row - num_filter_row, 0),
                                        (0, num_output_col - num_filter_col)), 
                                        "constant", constant_values=0)
        
        # create toeplitz matrixes from rows and add them to a list in reverse order
        toeplitz_matrixes = []
        for row in range(padded_filter.shape[0]-1, -1, -1):
            toeplitz_column = padded_filter[row].copy()
            toeplitz_row = np.r_[toeplitz_column[0], np.zeros(num_input_col-1)]

            matrix = toeplitz(c=toeplitz_column, r=toeplitz_row)
            toeplitz_matrixes.append(matrix)

        # get the indexes of the doubly-blocked matrix
        doubly_colums = range(1, padded_filter.shape[0]+1)
        doubly_rows = np.r_[doubly_colums[0], np.zeros(num_input_row-1, dtype=int)]

        double_indexes = toeplitz(doubly_colums, doubly_rows)

        # get the shape of the doubly-blocked matrix
        height = toeplitz_matrixes[0].shape[0]*double_indexes.shape[0]
        width = toeplitz_matrixes[0].shape[1]*double_indexes.shape[1]
        doubly_blocked_shape = (height, width)
        doubly_blocked = np.zeros(doubly_blocked_shape)

        # fill in the doubly blocked matrix
        block_height, block_width = toeplitz_matrixes[0].shape
        for row in range(double_indexes.shape[0]):
            for column in range(double_indexes.shape[1]):
                start_row = row * block_height
                start_column = column * block_width
                end_row = start_row + block_height
                end_column = start_column + block_width

                doubly_blocked[start_row:end_row, start_column:end_column] = toeplitz_matrixes[double_indexes[row, column]-1]
        
        return doubly_blocked, (num_output_row, num_output_col)

"""

cl  = ConvolutionalLayer(4, 2, 2)
cl.received_inputs.append(np.full((28, 28), 2))
image = np.ones((28, 28))

cl._backpropagation(image, 0.1)
"""
import numpy as np


class ConvolutionalLayer:

    def __init__(self, filter_size, stride_length, num_of_filters):
        self.filter_size = filter_size
        self.num_of_filters = num_of_filters
        self.stride_length = stride_length
        self._create_filters()

    def _create_filters(self):
        """Creates the filter that holds the weights.
        The weights are initialized to small random numbers.
        """
        self.filters = [0.01 * np.random.randn(self.filter_size,
                                        self.filter_size)
                        * np.sqrt(2.0 / 28*28)
                        for i in range(self.num_of_filters)]
        self.bias_vector = [0.01 for i in range(self.num_of_filters)]

    def convolute(self, image):

        self.received_inputs = image        
        self.conv_in_shape = image.shape
        _, input_dim, _ = image.shape
        output_dim = int((input_dim - self.filter_size) / self.stride_length) + 1

        output = np.zeros((self.num_of_filters, output_dim, output_dim))

        for filter in range(self.num_of_filters):
            img_y = out_y = 0
            while img_y + self.filter_size <= input_dim:
                img_x = out_x = 0
                while img_x + self.filter_size <= input_dim:
                    output[filter, out_y, out_x] = \
                    np.sum(self.filters[filter] * image[filter, img_y:img_y+self.filter_size, img_x:img_x+self.filter_size]) + self.bias_vector[filter]
                    img_x += self.stride_length
                    out_x += 1
                img_y += self.stride_length
                out_y += 1
        return output

    def update_parameters(self, batch_size, learning_rate, beta1, beta2, clip_threshold, iterations):
        """Updates the parameters with stored gradients
        from the backpropagation process.
        """
        for filter_i in range(len(self.filters)):
            # Gradient clipping
            self.gradient_filters[filter_i] = self._clip_gradient(self.gradient_filters[filter_i], clip_threshold)
            self.bias_gradients[filter_i] = self._clip_gradient(self.bias_gradients[filter_i], clip_threshold)
            # Update moment vectors
            self.filter_mean_grad[filter_i] = beta1*self.filter_mean_grad[filter_i] \
                + (1-beta1)*self.gradient_filters[filter_i]/batch_size
            self.filter_grad_variance[filter_i] = beta2*self.filter_grad_variance[filter_i] \
                + (1-beta2) * (self.gradient_filters[filter_i]/batch_size)**2
            # Take the bias-corrected variables
            filter_mhat = self.filter_mean_grad[filter_i] / \
                (1 - beta1**(iterations+1))
            filter_vhat = self.filter_grad_variance[filter_i] / (
                1 - beta2**(iterations+1))
            # Update variable
            self.filters[filter_i] -= learning_rate * \
                filter_mhat / (np.sqrt(filter_vhat)+1e-9)

            # Same for bias
            self.bias_mean_grad[filter_i] = beta1*self.bias_mean_grad[filter_i] \
                + (1-beta1)*self.bias_gradients[filter_i]/batch_size
            self.bias_grad_variance[filter_i] = beta2*self.bias_grad_variance[filter_i] \
                + (1-beta2) * (self.bias_gradients[filter_i]/batch_size)**2
            bias_mhat = self.bias_mean_grad[filter_i] / \
                (1 - beta1**(iterations+1))
            bias_vhat = self.bias_grad_variance[filter_i] / \
                (1 - beta2**(iterations+1))
            self.bias_vector[filter_i] -= learning_rate * \
                bias_mhat / (np.sqrt(bias_vhat)+1e-9)
    
    def _clip_gradient(self, gradient, clip_threshold):
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > clip_threshold:
            scaling_factor = clip_threshold / gradient_norm
            gradient *= scaling_factor
        return gradient            

    def initialize_gradients(self):
        """Initialize the gradients for
        the backpropagation process.
        The update follows the Adam optimization.
        """
        self.gradient_filters = np.zeros_like(self.filters)
        self.bias_gradients = np.zeros_like(self.bias_vector)

        self.filter_mean_grad = np.zeros_like(self.filters)
        self.filter_grad_variance = np.zeros_like(self.filters)

        self.bias_mean_grad = np.zeros_like(self.bias_vector)
        self.bias_grad_variance = np.zeros_like(self.bias_vector)


    def backpropagation(self, gradient_input, regularization_strength):
        """This function takes care of the backpropagation for the
        convolution layer. For each filter in the layer, it calls
        the _get_filter_gradient function to get the gradients for the filters
        and then updates the filters.
        """
        gradient_output = np.zeros(self.conv_in_shape)
        for filter_i in range(len(self.filters)):
            current_y = output_y = 0
            while current_y + self.filter_size <= self.conv_in_shape[1]:
                current_x = output_x = 0
                while current_x + self.filter_size <= self.conv_in_shape[2]:

                    self.gradient_filters[filter_i] += gradient_input[filter_i, output_y, output_x]\
                        * self.received_inputs[filter_i, current_y:current_y+self.filter_size,
                                               current_x:current_x+self.filter_size]
                    
                    gradient_output[filter_i, current_y:current_y+self.filter_size,
                                    current_x:current_x+self.filter_size]\
                        += gradient_input[filter_i, output_y, output_x] * self.filters[filter_i]
                    
                    current_x += self.stride_length
                    output_x += 1
                current_y += self.stride_length
                output_y += 1
            self.bias_gradients[filter_i] += np.sum(gradient_input[filter_i])

        return gradient_output
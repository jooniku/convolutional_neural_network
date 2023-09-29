from non_linearity import NonLinearity
from layers.pooling_layer import PoolingLayer
from layers.convolutional_layer import ConvolutionalLayer
from layers.fully_connected_layer import FullyConnectedLayer
from layers.input_layer import InputLayer

import numpy as np

class NeuralNetwork:
    """
    Class for the neural network. The layers are represented as classes also. 
    Hyperparameters are class objects. The NeuralNetwork class calls the
    layer classes and the main operations are performed in the layer classes.
    """

    def __init__(self):
        # hyperparameter initialization here
        # currently only kernel size 3 works
        self.kernel_size = 3 # meaning 3x3
        self.stride_length = 1
        self.num_of_convolution_layers = 1
        self.learning_step_size = 0.01
        self.regularization_strength = 0.001
        
        self._initialize_custom_functions()

    def _initialize_custom_functions(self):
        """Here are customisable functions,
        as in one can use max pooling or average pooling.
        Change those here.
        """

        self._get_training_data()
        self.non_linearity_function = NonLinearity()._relu
        self.pooling_function = PoolingLayer(self.kernel_size)._max_pooling
        self.fully_connected_layer = FullyConnectedLayer(self.learning_step_size, self.regularization_strength)
        self._create_convolutional_layers()
        
    def _get_training_data(self):
        """Imports the training data from the input layer
        so other layers can use it.
        """
        self.training_images, self.training_labels = InputLayer()._pass_training_data()


    def _create_convolutional_layers(self):
        """Creates all of the convolutional layers
        and adds them to a list where they can be referenced to.
        """

        self.convolutional_layers = []
        for i in range(self.num_of_convolution_layers):
            self.convolutional_layers.append(ConvolutionalLayer(self.kernel_size, self.stride_length))


    def _process_image(self, image: np.array):
        """For an image, add convolution, then non-linearity
        and finally pooling. After that, feed the image to the
        next convolutional layer and repeat.

        Args:
            image (np.array): _description_
        """
        for conv_layer in self.convolutional_layers:
            image = self._add_non_linearity(conv_layer._add_2d_convolution(raw_image=image))

        return image
    
    def _train_network(self, image: np.array, label: int):
        # in gradient descent loop
        # compute class probs
        # compute loss
        # compute gradient
        # backpropagate
        for conv_layer in self.convolutional_layers:
            convoluted_image = self._add_non_linearity(conv_layer._add_2d_convolution(raw_image=image))

        loss = self.fully_connected_layer._compute_loss(image=convoluted_image, kernel=conv_layer.kernel, label=label)
        gradients = self.fully_connected_layer._compute_gradient(image=convoluted_image, label=label)

        for conv_layer in range(len(self.convolutional_layers)-1, -1, -1):
            pass
            #self.convolutional_layers[conv_layer]._update_parameters(gradients, self.regularization_strength, self.learning_step_size)



    def _add_non_linearity(self, image: np.array):
        """This function takes the convoluted data and
        adds non-linearity with a non-linearity function specified 
        in the initiation method.

        Args:
            data (_type_): numpy array of an image

        Returns:
            _type_: Returns the modified data
        """

        """for each number in data call the non-linearity function
        and then return the modified data
        """
        for row in range(len(image)):
            for column in range(len(image[0])):
                image[row][column] = self.non_linearity_function(image[row][column])
        
        return image
        

    
    def _add_pooling(self, image: np.array):
        """This function takes the data and using a pooling
        algorithm specified in the initiation method, it 
        adds the pooling.

        Args:
            data (_type_): _description_
        
        Returns:
            _type_: Returns the pooled data.
        """

        """for each e.g. 3x3 section of data, feed it to the pooling algo
        pooling size should be defined here aswell, or mabye in the init method
        """
        return self.pooling_function(image=image)
    
    def _add_backpropagation(self):
        pass


from matplotlib import pyplot as plt

from layers.mnist_data_processor import training_images, training_labels

i = 2
nn = NeuralNetwork()
x = nn._process_image(image=training_images[i])
print(x)

plt.imshow(x, interpolation='nearest')
plt.show()
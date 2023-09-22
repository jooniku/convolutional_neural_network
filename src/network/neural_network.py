from src.network.non_linearity import NonLinearity
from src.network.layers.pooling_layer import PoolingLayer
from src.network.layers.convolutional_layer import ConvolutionalLayer
import numpy as np

class NeuralNetwork:
    """
    Class for the neural network. The layers are represented as subclasses. 
    Hyperparameters are class objects.
    """

    def __init__(self):
        # hyperparameter initialization here
        self.kernel_size = 3 # meaning 3x3
        self.num_of_convolution_layers = 2 
        
        self._initialize_custom_functions()

    def _initialize_custom_functions(self):
        """Here are customisable functions,
        as in one can use max pooling or average pooling.
        Change those here.
        """

        self.non_linearity_function = NonLinearity()._relu
        self.pooling_function = PoolingLayer(self.kernel_size)._max_pooling
        self._create_convolutional_layers()
        

    def _create_convolutional_layers(self):
        """Creates all of the convolutional layers
        and adds them to a list where they can be referenced to.
        """

        self.convolution_layers = []
        for i in range(self.num_of_convolution_layers):
            self.convolution_layers.append(ConvolutionalLayer(self.kernel_size))


    def _add_convolution(self, image: np.array):
        """For an image, run a convolution with all
        conv layers.

        Args:
            image (np.array): _description_
        """
        cl = ConvolutionalLayer(self.kernel_size)
        cl._add_2d_convolution(raw_image=image)



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
        

    
    def _add_pooling(self, data: list):
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
        pass


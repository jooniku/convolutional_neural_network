from non_linearity import NonLinearity
from layers.pooling_layer import PoolingLayer

class NeuralNetwork:
    """
    Class for the neural network. The layers are represented as subclasses. 
    Hyperparameters are class objects.
    """

    def __init__(self):
        # hyperparameter initialization here
        self.non_linearity_function = NonLinearity()._relu
        self.pooling_function = PoolingLayer()._max_pooling

    def _add_non_linearity(self, data):
        """This function takes the convoluted data and
        adds non-linearity with a non-linearity function specified 
        in the initiation method.

        Args:
            data (_type_): _description_

        Returns:
            _type_: Returns the data.
        """

        """for each number in data call the non-linearity function
        and then return the modified data
        """
        pass

    
    def _add_pooling(self, data):
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


nn = NeuralNetwork()

nn._add_non_linearity(-2)


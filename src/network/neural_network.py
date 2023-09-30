from non_linearity import NonLinearity
from layers.pooling_layer import PoolingLayer
from layers.convolutional_layer import ConvolutionalLayer
from layers.fully_connected_layer import FullyConnectedLayer
from layers.input_layer import InputLayer
from matplotlib import pyplot as plt

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
        self.num_of_convolution_layers = 2
        self.learning_step_size = 0.01
        self.regularization_strength = 0.001
        self.num_of_classes = 10 # as in 0...9
        
        self._initialize_custom_functions()

    def _initialize_custom_functions(self):
        """Here are customisable functions,
        as in one can use max pooling or average pooling.
        Change those here.
        """

        self._get_training_data()
        self.non_linearity_function = NonLinearity()._relu
        self.pooling_layer = PoolingLayer(self.kernel_size)._max_pooling
        self.fully_connected_layer = FullyConnectedLayer(self.num_of_classes, self.learning_step_size, self.regularization_strength)
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
    
    def _train_network(self):
        """This function is called to train the network.
        """
        for epoch in range(10):
            conv_images = []
            labels = []
            for data in range(196):
                image = self.training_images[data]
                label = self.training_labels[data]

                for conv_layer in self.convolutional_layers:
                   image = self.pooling_layer(self._add_non_linearity(conv_layer._add_2d_convolution(image=self._add_padding(image))))

                image = self.fully_connected_layer._process(image=image)
                conv_images.append(image)
                labels.append(label)
            
            loss = self.fully_connected_layer._compute_loss(images=conv_images, labels=labels)
            gradients = self.fully_connected_layer._compute_gradient(images=conv_images, labels=labels)

            gradient_weigth = self.fully_connected_layer._update_parameters(gradient_score=gradients)



            for conv_layer in range(len(self.convolutional_layers)-1, -1, -1):
                self.convolutional_layers[conv_layer]._update_parameters(loss, self.regularization_strength, self.learning_step_size)

            print(loss)            

    def _add_padding(self, image: np.array):
        """Adds zero-padding for the image to make sure the
        operations work.

        Args:
            image (np.array): array representation of image

        Returns:
            _type_: padded image
        """
        needed_padding = (32 - len(image)) // 2

        return np.pad(image, pad_width=needed_padding)


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
        return self.non_linearity_function(image)
        

    def _add_backpropagation(self):
        pass


nn = NeuralNetwork()

nn._train_network()
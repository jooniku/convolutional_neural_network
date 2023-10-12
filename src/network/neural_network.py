from network.non_linearity import NonLinearity
from network.layers.pooling_layer import PoolingLayer
from network.layers.convolutional_layer import ConvolutionalLayer
from network.layers.fully_connected_layer import FullyConnectedLayer
from network.layers.input_layer import InputLayer
import random
import numpy as np

class NeuralNetwork:
    """
    Class for the neural network. The layers are represented as classes also. 
    Hyperparameters are class objects. The NeuralNetwork class calls the
    layer classes and the main operations are performed in the layer classes.
    """

    def __init__(self, filter_size=3, # Note: (filter_size - stride_length) // 2 must return an integer
                 stride_length=1,     # Note: (W - F + 2P)//S + 1 also must return integer
                 num_of_convolutional_layers=1,
                 num_of_filters_in_conv_layer=2,
                 learning_step_size=0.01,
                 epochs=1000,
                 reg_strength=0.001,
                 batch_size=10,
                 num_of_classes=10):
        
        # hyperparameter initialization here
        self.filter_size = filter_size
        self.stride_length = stride_length
        self.num_of_convolution_layers = num_of_convolutional_layers
        self.num_of_filters_in_conv_layer = num_of_filters_in_conv_layer
        self.learning_step_size = learning_step_size
        self.epochs = epochs
        self.regularization_strength = reg_strength
        self.batch_size = batch_size
        self.num_of_classes = num_of_classes
        
        self._initialize_custom_functions()

    def _initialize_custom_functions(self):
        """Here are customisable functions,
        as in one can use max pooling or average pooling.
        Change those here.
        """

        self._get_training_data()
        self.non_linearity_function = NonLinearity()._relu
        self.pooling_layer = PoolingLayer(self.filter_size)
        self.pooling_function = self.pooling_layer._average_pooling
        self.fully_connected_layer = FullyConnectedLayer(self.num_of_classes, self.learning_step_size, self.regularization_strength)
        self._create_convolutional_layers()
        self._setup_training_batch()
        
    def _get_training_data(self):
        """Imports the training data from the input layer
        so other layers can use it.
        """
        self.training_images, self.training_labels = InputLayer()._pass_training_data()

    def _setup_training_batch(self):
        self.training_batch = [random.randint(0, 50000) for i in range(self.batch_size)]

    def _create_convolutional_layers(self):
        """Creates all of the convolutional layers
        and adds them to a list where they can be referenced to.
        """

        self.convolutional_layers = []
        for i in range(self.num_of_convolution_layers):
            self.convolutional_layers.append(ConvolutionalLayer(self.filter_size, self.stride_length, self.num_of_filters_in_conv_layer))


    def _predict(self, image: np.array):
        """For an image, add convolution, then non-linearity
        and finally pooling. After that, feed the image to the
        next convolutional layer and repeat.

        Args:
            image (np.array): _description_
        """
        for conv_layer in self.convolutional_layers:
            image = self.pooling_function(self._add_non_linearity(conv_layer._add_2d_convolution(image)))
        prediction = np.argmax(self.fully_connected_layer._process(image))

        return prediction
    
    def _train_network(self):
        """This function is called to train the network.
        """
        for epoch in range(self.epochs):
            conv_images = []
            labels = []
            for data in [random.randint(0, 50000) for i in range(self.batch_size)]:
                image = self.training_images[data]
                label = self.training_labels[data]

                for conv_layer in self.convolutional_layers:
                    image = self.pooling_function(self._add_non_linearity(conv_layer._add_2d_convolution(image)))

                image = self.fully_connected_layer._process(image=image)
                conv_images.append(image)
                labels.append(label)
                
            loss = self.fully_connected_layer._compute_average_loss(images=conv_images, labels=labels)
            gradients = self.fully_connected_layer._compute_average_gradient(images=conv_images, labels=labels)
                
                #loss = self.fully_connected_layer._compute_loss(image, label)
                #gradients = self.fully_connected_layer._compute_gradient(image, label)

            self._backpropagate_network(gradients=gradients, loss=loss)
            
            print("epoch:", epoch, "loss:", loss)
                

    def _backpropagate_network(self, gradients, loss):
        """This function takes care of the main backpropagation
        process. It goes through all of the layers and calls
        layer-specific backpropagation functions. 

        Args:
            gradients (_type_): gradient of the forward pass
            loss (_type_): loss of the forward pass
        """
        gradient_input = self.fully_connected_layer._update_parameters(gradient_score=gradients)

        # this is the shape that the convolutional layers take in
        output_shape = np.shape(self.convolutional_layers[0].received_inputs[0])

        for conv_layer in range(len(self.convolutional_layers)-1, -1, -1):
            gradient_input = self.pooling_layer._backpropagation_average_pooling(gradient_input, output_shape)
            gradient_input = self.convolutional_layers[conv_layer]._backpropagation(gradient_input, self.learning_step_size)



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
        
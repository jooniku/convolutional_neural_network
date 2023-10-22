from src.network.non_linearity import NonLinearity
from src.network.layers.pooling_layer import PoolingLayer
from src.network.layers.convolutional_layer import ConvolutionalLayer
from src.network.layers.fully_connected_layer import FullyConnectedLayer
from src.network.layers.input_layer import InputLayer
import random
import numpy as np


class NeuralNetwork:
    """
    Class for the neural network. The layers are represented as classes also. 
    Hyperparameters are class objects. The NeuralNetwork class calls the
    layer classes and the main operations are performed in the layer classes.
    """

    def __init__(self, filter_size=3,
                 stride_length=2,
                 num_of_convolutional_layers=2,
                 num_of_filters_in_conv_layer=5,
                 learning_step_size=0.01,
                 epochs=2000,
                 reg_strength=0,
                 batch_size=1,
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
        self.pooling_layer = PoolingLayer(kernel_size=2)
        self.pooling_function = self.pooling_layer._average_pooling
        self.fully_connected_layer = FullyConnectedLayer(
            self.num_of_classes, self.learning_step_size, self.regularization_strength)
        self._create_convolutional_layers()

    def _get_training_data(self):
        """Imports the training data from the input layer
        so other layers can use it.
        """
        training_images, training_labels = InputLayer()._pass_training_data()
        self.training_data = np.hstack((training_images, training_labels))
        


    def _create_convolutional_layers(self):
        """Creates all of the convolutional layers
        and adds them to a list where they can be referenced to.
        """

        self.convolutional_layers = []
        for i in range(self.num_of_convolution_layers):
            self.convolutional_layers.append(ConvolutionalLayer(
                self.filter_size, self.stride_length, self.num_of_filters_in_conv_layer))

    def _predict(self, image: np.array):
        """For an image, add convolution, then non-linearity
        and finally pooling. After that, feed the image to the
        next convolutional layer and repeat.

        Args:
            image (np.array): _description_
        """
        images = []
        for i in range(self.num_of_filters_in_conv_layer):
            images.append(image)
        images = np.array(images)
        for conv_layer in self.convolutional_layers:
            images = self.pooling_function(self._add_non_linearity(
                conv_layer._add_2d_convolution(images)))
        
        prediction = np.argmax(self.fully_connected_layer._process(images))
        return prediction

    def _train_network(self):
        """This function is called to train the network.
        """
        for epoch in range(self.epochs):
            gradients = np.zeros((self.num_of_classes))
            loss = 0
            np.random.shuffle(self.training_data)
            batches = [self.training_data[i: i + self.batch_size] for i in range(0, self.training_data.shape[0], self.batch_size)]

            for batch in batches:
                batch_images = batch[:, 0:-1]
                batch_images = batch_images.reshape(len(batch), 28, 28)
                labels = batch[:, -1]
                # stochastic gradient descent
                for data in range(len(batch_images)):
                    label = int(labels[data])

                    images = []
                    for i in range(self.num_of_filters_in_conv_layer):
                        images.append(batch_images[data])
                    images = np.array(images)

                    #for conv_layer in self.convolutional_layers:
                        #images = self.pooling_function(self._add_non_linearity(
                        #    conv_layer._add_2d_convolution(images)))

                    images = self.pooling_function(self._add_non_linearity(self.convolutional_layers[0]._add_2d_convolution(images)))
                    images = self.pooling_function(self.convolutional_layers[1]._add_2d_convolution(images))

                    probs = self.fully_connected_layer._process(images=images)
                    print(probs)

                    loss += self.fully_connected_layer._compute_loss(
                        probabilities=probs, label=label)
                    gradients += self.fully_connected_layer._compute_gradient(
                        probabilities=probs, label=label)

                gradients /= self.batch_size
                loss /= self.batch_size
                self._backpropagate_network(gradients=gradients)

            print("epoch:", epoch, "loss:", loss)

            if epoch % 2000 == 0:
                self.learning_step_size *= 0.9
            if loss < 0.1:
                break


    def _backpropagate_network(self, gradients):
        """This function takes care of the main backpropagation
        process. It goes through all of the layers and calls
        layer-specific backpropagation functions. 

        Args:
            gradients (_type_): gradient of the forward pass
            loss (_type_): loss of the forward pass
        """
        gradient_input = self.fully_connected_layer._update_parameters(
            gradient_score=gradients)

        #for conv_layer in range(len(self.convolutional_layers)-1, -1, -1):
        output_shape = self.convolutional_layers[1].received_inputs.shape[1]

        gradient_input = self.pooling_layer._backpropagation_average_pooling(
            gradient_input, output_shape)

        gradient_input = self.non_linearity_function(gradient_input)

        gradient_input = self.convolutional_layers[1]._backpropagation(
            gradient_input, self.learning_step_size, self.regularization_strength)
        
        output_shape = self.convolutional_layers[0].received_inputs.shape[1]

        gradient_input = self.pooling_layer._backpropagation_average_pooling(
            gradient_input, output_shape)

        gradient_input = self.convolutional_layers[0]._backpropagation(
            gradient_input, self.learning_step_size, self.regularization_strength)

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

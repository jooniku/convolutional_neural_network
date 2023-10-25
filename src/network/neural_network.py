from src.network.non_linearity import NonLinearity
from src.network.layers.pooling_layer import PoolingLayer
from src.network.layers.convolutional_layer import ConvolutionalLayer
from src.network.layers.fully_connected_layer import FullyConnectedLayer
from src.network.layers.input_layer import InputLayer
from src.network.layers.classifier import Classifier
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pathlib
import os
from tqdm import tqdm


class NeuralNetwork:
    """
    Main class for the convolutional neural network. 
    Layers are represented as classes and this class
    calls layer classes. 
    """

    def __init__(self, filter_size=3,
                 stride_length=2,
                 num_of_convolutional_layers=2,
                 num_of_filters_in_conv_layer=8,
                 learning_rate=0.0008,
                 epochs=2,
                 reg_strength=0.001,
                 batch_size=64,
                 num_of_classes=10):

        # hyperparameter initialization here
        self.filter_size = filter_size
        self.stride_length = stride_length
        self.num_of_convolution_layers = num_of_convolutional_layers
        self.num_of_filters_in_conv_layer = num_of_filters_in_conv_layer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization_strength = reg_strength
        self.batch_size = batch_size
        self.num_of_classes = num_of_classes

        self._initialize_custom_functions()

    def _initialize_custom_functions(self):
        """Here are customisable functions,
        as in one can use max-pooling or average pooling.
        Change those here.
        """

        self._get_training_data()

        self.non_linearity_function = NonLinearity()._relu
        self.pooling_layer = PoolingLayer(kernel_size=2)
        self.fully_connected_layer = FullyConnectedLayer(
            self.num_of_classes, (8, 12, 12))
        self.classifier = Classifier()

        self._create_convolutional_layers()

    def load_latest_saved_network(self):
        """This function loads the latest trained
        network to use. Works only with current architecture.
        """
        dir_path = pathlib.Path("data/trained_networks").resolve()
        all_saved_networks = os.listdir(dir_path)

        latest_file = max(all_saved_networks, key=str)

        latest_file_path = str(dir_path) + "/" + latest_file

        saved_data = np.load(latest_file_path)

        self.convolutional_layers[0].filters = saved_data["filters1"]
        self.convolutional_layers[0].bias_vector = saved_data["biases1"]

        self.convolutional_layers[1].filters = saved_data["filters2"]
        self.convolutional_layers[1].bias_vector = saved_data["biases2"]

        self.fully_connected_layer.weight_matrix = saved_data["fc_weight"]
        self.fully_connected_layer.bias = saved_data["fc_bias"]

        print("Network loaded successfully")

    def _save_network(self):
        """Saves the current network to a file.
        Gives the current date and time as filename.
        """
        path = pathlib.Path("data/trained_networks")
        absolute_path = path.resolve()
        file_name = str(absolute_path) + "/" + \
            datetime.now().strftime('%d-%m-%Y-%H-%M')

        np.savez(file_name, filters1=self.convolutional_layers[0].filters,
                 biases1=self.convolutional_layers[0].bias_vector,
                 filters2=self.convolutional_layers[1].filters,
                 biases2=self.convolutional_layers[1].bias_vector,
                 fc_weight=self.fully_connected_layer.weight_matrix,
                 fc_bias=self.fully_connected_layer.bias)

        print("Network saved successfully")

    def _get_training_data(self):
        """Imports the training data from the input layer
        so other layers can use it.
        """
        self.training_images, self.training_labels = InputLayer().pass_training_data()
        self.training_data = np.hstack(
            (self.training_images, self.training_labels))

    def _create_convolutional_layers(self):
        """Creates all of the convolutional layers
        and adds them to a list where they can be referenced to.
        """

        self.convolutional_layers = []
        for i in range(self.num_of_convolution_layers):
            self.convolutional_layers.append(ConvolutionalLayer(
                self.filter_size, self.stride_length,
                self.num_of_filters_in_conv_layer))

    def predict(self, image):
        """This function is called to make 
        the prediction of the class for an image.
        It feeds the image to the network and returns
        the position with the largest probability.
        """
        images = np.array([image for i in range(
            self.num_of_filters_in_conv_layer)])
        probabilities = self._feedforward(images)
        prediction = np.argmax(probabilities)

        return prediction

    def _feedforward(self, images):
        """This is the function to 
        feed the input image through the
        network. It returns the probability
        distribution of the classes.
        """
        for conv_layer in self.convolutional_layers:
            images = self.non_linearity_function(
                conv_layer.add_2d_convolution(images))

        images = self.pooling_layer.average_pooling(images)

        images = self.fully_connected_layer.process(images=images)

        probs = self.classifier.compute_probabilities(images)

        return probs

    def _gradient_descend(self, batch_images, labels):
        """This is the gradient descent
        algorithm. It takes a batch and for each
        image in the batch it feeds the image through
        the network and stores it's gradients and 
        calculates it's loss. 
        """
        loss = 0
        for data in range(len(batch_images)):
            label = int(labels[data])

            images = np.array([batch_images[data]
                               for i in range(self.num_of_filters_in_conv_layer)])

            probabilities = self._feedforward(images=images)

            loss += self.classifier.compute_loss(
                probabilities=probabilities, label=label)
            gradients = self.classifier.compute_gradients(
                probabilities=probabilities, label=label)

            self._backpropagate_network(gradients=gradients)

        average_loss = loss / len(batch_images)

        return average_loss

    def train_network(self, save_network=True):
        """This is the main function in 
        training the neural network. For each 
        pass on the training data, it splits the data 
        into batches, computes gradients and updates weights
        with stochastic gradient descent algorithm.
        """
        self._initialize_loss_plot()
        batchcount = 0

        for epoch in range(self.epochs):
            np.random.shuffle(self.training_data)
            batches = [self.training_data[i: i + self.batch_size]
                       for i in range(0, self.training_data.shape[0], self.batch_size)]

            progress = tqdm(batches)
            for _, batch in enumerate(progress):
                # separate images and labels and reshape back to normal
                batch_images = batch[:, 0:-1]
                batch_images = batch_images.reshape(len(batch), 28, 28)
                labels = batch[:, -1]

                self._initialize_gradients()

                average_loss = self._gradient_descend(
                    batch_images=batch_images, labels=labels)

                self._update_network_parameters()

                self.cost.append(average_loss)
                self.batch_values.append(batchcount)
                batchcount += 1
                progress.set_description("Cost: %.2f" % (self.cost[-1]))

                self._plot_loss()

            print("epoch:", epoch)
            self.learning_rate *= 0.5
        print("Network training complete")

        if save_network:
            self._save_network()

    def _initialize_loss_plot(self):
        """Initialize values for plotting
        of loss.
        """
        self.cost = []
        self.batch_values = []

        plt.ion()
        self.fig, self.axis = plt.subplots()

    def _plot_loss(self):
        """This function plots
        the loss on the y-self.axis and
        batches iterated on the x-self.axis.
        This provides live information
        about the progress of the training.
        """
        self.axis.clear()
        self.axis.plot(self.batch_values, self.cost,
                       label='Training Loss')
        self.axis.set_title('Live Training Loss')
        self.axis.set_xlabel('Batches')
        self.axis.set_ylabel('Cost')
        self.axis.legend()
        plt.pause(0.01)

    def _initialize_gradients(self):
        """This function calls all of the layers to initialize
        their gradients. Is called for each batch.
        """
        self.fully_connected_layer.initialize_gradients()
        for conv_layer in range(len(self.convolutional_layers)):
            self.convolutional_layers[conv_layer].initialize_gradients()

    def _update_network_parameters(self):
        """This function calls each layer to update
        its parameters with the gradients it has stored.
        The update method is using adam-momentum.
        """
        self.fully_connected_layer.update_parameters(batch_size=self.batch_size,
                                                     learning_rate=self.learning_rate)

        for conv_layer in range(len(self.convolutional_layers)):
            self.convolutional_layers[conv_layer].update_parameters(batch_size=self.batch_size,
                                                                    learning_rate=self.learning_rate)

    def _backpropagate_network(self, gradients):
        """This function takes care of the main backpropagation
        process. It goes through all of the layers and calls
        layer-specific backpropagation functions. For every layer
        it gives the gradient from the previous layer.
        """
        gradient_input = self.fully_connected_layer.backpropagation(
            gradient_score=gradients, reg_strength=self.regularization_strength)
        output_shape = self.convolutional_layers[-1].received_inputs.shape[1]

        gradient_input = self.pooling_layer.backpropagation_average_pooling(
            gradient_input, output_shape)

        for conv_layer in range(len(self.convolutional_layers)-1, -1, -1):
            gradient_input = self.non_linearity_function(gradient_input)
            gradient_input = self.convolutional_layers[conv_layer].backpropagation(
                gradient_input, self.regularization_strength)

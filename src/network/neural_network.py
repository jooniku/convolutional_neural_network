import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pathlib
import os
from tqdm import tqdm
from src.network.non_linearity import NonLinearity
from src.network.layers.pooling_layer import PoolingLayer
from src.network.layers.convolutional_layer import ConvolutionalLayer
from src.network.layers.fully_connected_layer import FullyConnectedLayer
from src.network.layers.input_layer import InputLayer
from src.network.layers.classifier import Classifier


class NeuralNetwork:
    """
    Main class for the convolutional neural network. 
    Layers are represented as classes and this class
    calls layer classes. 
    """

    def __init__(self, filter_size=5,
                 stride_length=1,
                 num_of_convolutional_layers=3,
                 num_of_filters_in_conv_layer=15,
                 learning_rate=1e-2,
                 epochs=3,
                 reg_strength=0,
                 batch_size=250,
                 num_of_classes=10,
                 beta1=0.95,
                 beta2=0.99):

        self.filter_size = filter_size
        self.stride_length = stride_length
        self.num_of_convolution_layers = num_of_convolutional_layers
        self.num_of_filters_in_conv_layer = num_of_filters_in_conv_layer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization_strength = reg_strength
        self.batch_size = batch_size
        self.num_of_classes = num_of_classes
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate_schedule = []

        self._initialize_custom_functions()

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
            images = conv_layer.convolute(images)
            images = self.non_linear.forward(images, "conv_layer")

        images = self.pooling_layer.max_pooling(images)
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
        self._initialize_plots()
        self.iterations = 1
        val_accuracy = 0
        for epoch in range(1, self.epochs+1):
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

                if self.iterations % 3 == 0:
                    val_accuracy = self._test_validation_accuracy()

                    # save network incase overfitting
                    #if val_accuracy > 98 or average_loss < 0.1:
                    #    self._save_network()
                    #    self._save_plots()

                self.loss_values.append(average_loss)
                self.batch_values.append(self.iterations)
                self.validation_accuracy.append(val_accuracy)

                progress.set_description("Loss: %.2f" % (self.loss_values[-1]))
                self._plot_data()
                self.iterations += 1

                if self.iterations in self.learning_rate_schedule:
                    self.learning_rate *= 0.5
            print("epoch:", epoch)
        self._stop_training(save_network)

    def _update_network_parameters(self):
        """This function calls each layer to update
        its parameters with the gradients it has stored.
        The update method is using adam-momentum.
        """
        clip_threshold = np.inf
        self.fully_connected_layer.update_parameters(batch_size=self.batch_size,
                                                     learning_rate=self.learning_rate,
                                                     beta1=self.beta1,
                                                     beta2=self.beta2,
                                                     clip_threshold=clip_threshold,
                                                     iterations=self.iterations)

        for conv_layer in range(len(self.convolutional_layers)):
            self.convolutional_layers[conv_layer].update_parameters(batch_size=self.batch_size,
                                                                    learning_rate=self.learning_rate,
                                                                    beta1=self.beta1,
                                                                    beta2=self.beta2,
                                                                    clip_threshold=clip_threshold,
                                                                    iterations=self.iterations)

    def _backpropagate_network(self, gradients):
        """This function takes care of the main backpropagation
        process. It goes through all of the layers and calls
        layer-specific backpropagation functions. For every layer
        it gives the gradient from the previous layer.
        """

        gradient_input = self.fully_connected_layer.backpropagation(
            gradient_score=gradients, reg_strength=self.regularization_strength)

        output_shape = 16

        gradient_input = self.pooling_layer.max_pooling_backpropagation(
            gradient_input, output_shape)

        for conv_layer in range(len(self.convolutional_layers)-1, -1, -1):
            gradient_input = self.non_linear.backpropagation(gradient_input,
                                                                  "conv_layer",
                                                                    conv_layer)
            gradient_input = self.convolutional_layers[conv_layer].backpropagation(
                gradient_input, self.regularization_strength)

    def _test_validation_accuracy(self):
        """This function runs a validation
        check, where the model is tested with
        a small (40) number of unseen examples.
        """
        correct_predictions = 0
        for i in range(len(self.validation_images)):
            prediction = self.predict(self.validation_images[i])
            if prediction == self.validation_labels[i]:
                correct_predictions += 1

        return (correct_predictions/len(self.validation_images))*100

    def _initialize_custom_functions(self):
        """Here are customisable functions,
        as in one can use max-pooling or average pooling.
        Change those here.
        """

        self._get_training_data()
        self._get_validation_data()

        self.non_linear = NonLinearity()
        self.pooling_layer = PoolingLayer(kernel_size=2, stride=2)
        self.fully_connected_layer = FullyConnectedLayer(
            self.num_of_classes, (self.num_of_filters_in_conv_layer, 8, 8),
            self.non_linear)
        self.classifier = Classifier()

        self._create_convolutional_layers()

    def load_saved_network(self, load_latest=True, filename=None):
        """This function loads the latest trained
        network to use. Works only with current architecture.
        """
        dir_path = pathlib.Path("data/trained_networks").resolve()
        all_saved_networks = os.listdir(dir_path)

        if load_latest:
            latest_file = max(all_saved_networks, key=str)
        else:
            if filename is None:
                print("You have to specify a file")
                return
            latest_file = filename

        latest_file_path = str(dir_path) + "/" + latest_file

        saved_data = np.load(latest_file_path)

        self._load_hyperparameters(saved_data["hyperparameters"])

        self.convolutional_layers[0].filters = saved_data["filters1"]
        self.convolutional_layers[0].bias_vector = saved_data["biases1"]

        self.convolutional_layers[1].filters = saved_data["filters2"]
        self.convolutional_layers[1].bias_vector = saved_data["biases2"]

        self.fully_connected_layer.weight_matrixes[0] = saved_data["fc_weight1"]
        self.fully_connected_layer.biases[0] = saved_data["fc_bias1"]

        self.fully_connected_layer.weight_matrixes[1] = saved_data["fc_weight2"]
        self.fully_connected_layer.biases[1] = saved_data["fc_bias2"]

        print("Network loaded successfully")

    def _save_network(self):
        """Saves the current network to a file.
        Gives the current date and time as filename.
        """
        path = pathlib.Path("data/trained_networks")
        absolute_path = path.resolve()
        file_name = str(absolute_path) + "/" + \
            datetime.now().strftime('%d-%m-%Y-%H-%M')

        hyperparams = [self.filter_size,
                       self.stride_length,
                       self.num_of_convolution_layers,
                       self.num_of_filters_in_conv_layer,
                       self.learning_rate, self.epochs,
                       self.regularization_strength, self.batch_size,
                       self.num_of_classes, self.beta1,
                       self.beta2]

        np.savez(file_name, filters1=self.convolutional_layers[0].filters,
                 biases1=self.convolutional_layers[0].bias_vector,
                 filters2=self.convolutional_layers[1].filters,
                 biases2=self.convolutional_layers[1].bias_vector,
                 fc_weight1=self.fully_connected_layer.weight_matrixes[0],
                 fc_bias1=self.fully_connected_layer.biases[0],
                 fc_weight2=self.fully_connected_layer.weight_matrixes[1],
                 fc_bias2=self.fully_connected_layer.biases[1],
                 hyperparameters=hyperparams)

        print("Network saved successfully")

    def _load_hyperparameters(self, hyperparameters):
        """Loads hyperparameters from savefile.
        """
        self.filter_size = int(hyperparameters[0])
        self.stride_length = int(hyperparameters[1])
        self.num_of_convolution_layers = int(hyperparameters[2])
        self.num_of_filters_in_conv_layer = int(hyperparameters[3])
        self.learning_rate = float(hyperparameters[4])
        self.epochs = int(hyperparameters[5])
        self.regularization_strength = float(hyperparameters[6])
        self.batch_size = int(hyperparameters[7])
        self.num_of_classes = int(hyperparameters[8])
        self.beta1 = float(hyperparameters[9])
        self.beta2 = float(hyperparameters[10])

        # to get correct structure of network
        self._initialize_custom_functions()

    def _stop_training(self, save_network):
        print("Network training complete")
        print("Minimum loss achieved:", np.amin(self.loss_values))

        if save_network:
            self._save_network()
            self._save_plots()

    def get_test_data(self):
        """Passes the test data.
        """
        return InputLayer().pass_test_data()

    def _get_training_data(self):
        """Imports the training data from the input layer
        so other layers can use it.
        """
        self.training_data = InputLayer().pass_training_data()

    def _get_validation_data(self):
        """Import the validation data from the input layer.
        """
        self.validation_images, self.validation_labels = InputLayer().pass_validation_data()

    def _create_convolutional_layers(self):
        """Creates all of the convolutional layers
        and adds them to a list where they can be referenced to.
        """

        self.convolutional_layers = []
        for i in range(self.num_of_convolution_layers):
            self.convolutional_layers.append(ConvolutionalLayer(
                self.filter_size, self.stride_length,
                self.num_of_filters_in_conv_layer))

    def get_layer_activations(self):
        """This function gets the
        activation from each layer
        for visualization of the network
        functionality.
        """
        input_image = self.validation_images[np.random.randint(
            0, len(self.validation_images))]
        input_image = np.array([input_image
                               for i in range(self.num_of_filters_in_conv_layer)])
        activations = [input_image]

        for conv_layer in self.convolutional_layers:
            input_image = conv_layer.convolute(input_image)
            activations.append(input_image)
            input_image = self.non_linear.forward(input_image, "conv_layer")
            activations.append(input_image)

        input_image = self.pooling_layer.max_pooling(input_image)
        activations.append(input_image)
        input_image = self.fully_connected_layer.process(input_image)
        probs = self.classifier.compute_probabilities(input_image)

        self._visualize_activation_maps(activations)
        self._visualize_probability_distribution(probs)

    def _visualize_activation_maps(self, activations):
        """Visualizes the activation maps produced.
        """
        titles = ["Input Image", "Convolutional Layer",
                  "Leaky ReLU", "Convolutional Layer",
                  "Leaky ReLU", "Max Pooling"]

        # Plot layers
        plt.figure(figsize=(10, 10))
        plt.title("Layer Activations", pad=30)
        num_filters = len(activations[0])
        for layer in range(len(activations)):
            plt.subplot(1, len(activations), layer+1)
            for filter in range(num_filters):
                plt.subplot(num_filters, len(activations),
                            filter*len(activations)+layer+1)
                plt.imshow(activations[layer][filter], cmap="viridis")
                plt.axis("off")
            plt.title(titles[layer], y=-1)
        plt.tight_layout()
        plt.show()

    def _visualize_probability_distribution(self, probabilities):
        labels = [i for i in range(self.num_of_classes)]
        plt.bar(labels, probabilities*100, tick_label=labels)
        plt.xlabel("Class Label")
        plt.ylabel("Probability (%)")
        plt.title("Probability Distribution")
        plt.show()

    def _save_plots(self):
        """Save plots as images.
        """
        path = pathlib.Path("data/training_data_images")
        absolute_path = path.resolve()
        file_name = str(absolute_path) + "/" + \
            datetime.now().strftime('%d-%m-%Y-%H-%M')

        plt.savefig(file_name)

    def _initialize_plots(self):
        """Initialize values for plotting
        of loss and validation accuracy.
        """
        self.loss_values = []
        self.batch_values = []

        self.validation_accuracy = []

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1,
                                                      figsize=(8, 10),
                                                      dpi=100)

        self.ax1.set_xlabel("Batches")
        self.ax1.set_ylabel("Loss")
        self.ax1.set_title("Training Loss")

        self.ax2.set_xlabel("Batches")
        self.ax2.set_ylabel("Accuracy (%)")
        self.ax2.set_title("Validation Accuracy")

    def _plot_data(self):
        """This function plots
        live data about loss and
        validation accuracy.
        """

        line1 = self.ax1.plot(self.batch_values,
                              self.loss_values,
                              color="b")
        line2 = self.ax2.plot(self.batch_values,
                              self.validation_accuracy,
                              color="r")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _initialize_gradients(self):
        """This function calls all of the layers to initialize
        their gradients. Is called for each batch.
        """
        self.fully_connected_layer.initialize_gradients()
        for conv_layer in range(len(self.convolutional_layers)):
            self.convolutional_layers[conv_layer].initialize_gradients()

# Implementation documentation

## Project structure
- User interacts with the main application
- Dataset to be used can be selected (currently just the MNIST dataset)
- User will not have direct access to the workings of the network and all inner functions/methods are hidden
- User can train a new network or test a pre-trained model

## Use of LLMs
- ChatGPT3.5 has been used to help understand some concepts regarding backpropagation, plotting data and testing etc.
- All code is written by myself
- Architecture is designed and implemented by myself

## Sources
Here are the main sources for the project
- [Convolutional Neural Networks from the ground up](https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1)
- [Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)

## Architecture of the CNN
- Has an input layer, two convolutional layers with non-linearity functions and a pooling layer proceeded by 2 fully connected layers using softmax

![Visualized architecture](https://github.com/jooniku/digit_recognition_project/blob/main/Documentation/images/cnn_architecture.png)

### Input layer
- Takes the set of images as numpy arrays and labels as a numpy array
- Feeds it to the network

### Convolutional layer
- No padding is applied
- Convolutes the image by taking a filter and sliding it across the image and taking a dot-product and then a sum of weights and local image. Then adds a bias.
- The weight matrix (kernel) and bias vectors are initialized with values suggested in the Stanford course material of best current practises
- Weigth initialization formula is _w = np.random.randn(n) * sqrt(2.0/n)_, bias vector has all values of _0.01_

### Rectified Linear Unit
- Adds non-linearity to the network.
- In this network a "Leaky"-ReLU is used.
- If a given value in image is less than 0, the value becomes 0.0001*original_value

### Pooling layer
- Max pooling takes the largest value of the kernels local area and reduces size to 1 value

###  Fully connected layer
- Has an individual weight for each value of input.
- Weights and bias are initialized in the same way as with the convolutional layer.
- Reduces the input to a vector with a length of the amount of classes the dataset has. In the case of MNIST-dataset, 10.
- This layer's computation is done with matrix multiplication.

### Classifier layer
- Final layer of the network.
- Calculates the probabilities of each class based on the output of the FC-layer.
- Probabilities are calculated with Softmax function.
- Each value represents the probability of the class of it's index. E.g. Array [0.05, 0.8,..., 0.001] has the probability of 0.8 for number "1".
- Also computes cross-entropy loss and gradients

## Training the network
- A short description of the inner-workings of the network when training
- The training data is processed by the input layer
- Some data preprocessing is done, such as reducing mean and division with standard deviation
- Data is processed into batches for improved learning

### Gradient descent algorithm
- This loop is called for each batch of images in the dataset.
- For each image, do a feedforward of the image in the network to get a probability distribution
- Compute loss from the probability distribution. Loss is a metric of how wrong the probability distribution is. The main goal is to minimize loss.
- Compute the gradients from the probability distribution. The gradients represent the direction which reduces loss.
- Send the gradients to each layer through backpropagation

### Feedforward
- This is the main workhorse of the network, it is called for each image in the dataset.
- In this layer we send the input image through the network like I have described in the architecture part of this documentation.

### Backpropagation
- Backpropagation is the most important part of the neural network, without it, it couldn't learn.
- Essentially, the point is to change each weight in the network to become more accurate at predicting the correct class.
- In more detail, we take the gradient and compute it with our input image to the layer. This gives us the gradients specific to the filter we are dealing with.
- Then we reduce a proportion (with learning rate) of the gradients from the current weights. We reduce the sum of the gradients from the bias.
- Finally we compute an output gradient from the input gradient and the weights

### Learning
- These 3 preceding parts are done for a specified number of iterations. Slowly the loss converges close to zero.
- After we are satisfied with the "amount of learning" of the network, we can save the parameters to a file

## Predicting
- Now that the network is functional and accurate at predicting information from images, we can test it.
- To predict an image, we use the feedforward loop of the network to get the probability distribution.
- Then simply choose the largest probability and the index of that number is our prediction.


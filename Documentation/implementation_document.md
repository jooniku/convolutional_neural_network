## Architecture Implementation
Images are preprocessed with an external library before introducing them to the network.

The network will have 5 layers:
A convolutional layer followed by a pooling layer which is then repeated once and followed by a fully connected layer. Lastly a softmax regression model is used for classification.

### Convolutional layer
In this layer, multiple feature detectors (matrixes) are applied to the preprocessed image and a feature map is created. 

The values of the feature detectors are determined during the training phase with the use of a backpropagation algorithm.

After this non-linearity will be increased with a rectified linear unit (ReLU) function.

Padding is introduced to increase information retention with subsequent convolutions.

### Pooling layer
The pooling layer reduces the information complexity of the image with an average pooling algorithm, resulting in faster computation. 

### Fully connected layer
The fully connected layer connects to every node on the final pooling layer and introduces softmax regression.

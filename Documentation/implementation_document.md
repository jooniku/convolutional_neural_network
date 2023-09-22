# Implementation documentation

## Project structure
- User interacts with the main application
- Dataset to be used can be selected (currently just the MNIST dataset)
- Later user can customize the neural network e.g. size of kernels, pooling functions etc.
- User will not have direct access to the workings of the network and all inner functions/methods are hidden

## Architecture of the CNN
- Based on the LeNet-5 architecture
- Has an input layer, convolutional layer with non-linearity function and a pooling layer proceeded by a fully connected layer using softmax

### Input layer
- Takes the set of images as numpy arrays and labels as a numpy array
- Feeds it to the first convolutional layer

### Convolutional layer
- Checks size of the image and adds appropriate padding. The amount of padding is calculated with Padding = (kernel width - stride length) // 2 as suggested in the Stanford course material
- Reshapes the image array and does matrix multiplication with the kernel and adds a bias vector, then reshapes back
- The weight matrix (kernel) and bias vectors are initialized with values suggested in the Stanford course material of best current practises
- Weigth initialization formula is _w = np.random.randn(n) * sqrt(2.0/n)_, bias vector has all values of _0.01_

### Pooling layer
- Has currently two different pooling functions
- Average pooling takes an average of all the values within a kernels local space (e.g. 3x3 area) and reduces the size to 1 value
- Max pooling takes the largest value of the kernels local area and reduces size to 1 value
- Convolutional layer then adds appropriate padding to reduce information loss

  ###  Fully connected layer
  - Not implemented yet
 
  

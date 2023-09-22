This week was mainly spent writing code and some tests, roughly 10 hours.

Currently the network creates appropriate layers and calls them to use customizable functions such that e.g. different pooling functions can be used.

The MNIST dataset is now imported and the network can use the images/labels as a numpy array. ReLU and zero-padding functions work properly. Convolution is going to work next week, it should use a reshaping of the numpy array, matrix calculation and then reshaping it back. The input layer takes in preprocessed data and later can be used for other datasets as well, as customizability is one of the main focuses in this project. 

Testing document and the implementation document have been started and test coverage is visible. The test coverage report shows 88%, which is not correct, since there are functions which are not at the stage to be tested yet. The ones that are, have tests.

Due to a heavy workload from other courses I was not able to write many functions, but progression gets faster during upcoming weeks.

Questions:
Is it necessary to write time- and space complexities for the functions, as I'm writing them myself and not using any known algorithms?

# Requirement specification

## Problem specification
The idea is to create a tool that can take an image of hand-written numbers which are then recognized and classified appropriately.

## Solution specification
The solution is implemented with a convolutional neural network (CNN) trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset of hand-written digits.

The neural network follows the LeNet-5 architecture and for classification, a softmax regression model is used to return probabilities for each digit 0-9.

The LeNet-5 architecture is simple enough to implement in a relatively short timeframe with decent accuracy of classification (hopefully). 

An external library will likely be used for the preprocessing of images such as greyscale normalization and resizing. 

Numpy is used for calculations.

Data-structures used are mainly lists, nested lists, tuples etc.

Algorithms used are at least a backpropagation algorithm, an average pooling algorithm, different sliding window algorithms with matrix calculations and a softmax regression algorithm.

It is very difficult to give any estimates on time and space complexity, since there are no immediatelly avaliable well-known algorithms for these (at least ones that I was able to find). These estimates can be added later, if necessary.

## Sources
- _"Handwritten Digit Recognition System Based on Convolutional Neural Network"_, J. Li, et al. 2020 [IEEE](https://ieeexplore.ieee.org/document/9213619)
- _"Fast learning algorithm for deep belief nets*"_  [Geoffrey E. Hinton et al.](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)
- [The Deep Learning textbook](https://www.deeplearningbook.org/) was used as a learning resource
- [Convolutional Neural Network (CNN) Tutorial](https://www.kaggle.com/code/kanncaa1/convolutional-neural-network-cnn-tutorial/notebook)
- [Build your own neural network in R](https://www.kaggle.com/code/russwill/build-your-own-neural-network-in-r)
- [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
- [Softmax function](https://en.wikipedia.org/wiki/Softmax_function)

## Specific to the course
- I am in the bachelor's of computer science programme (TKT)
- Currently I only understand python well enough to give meaningful critique
- The language for the complete project is English

# User Manual
Once the project has been installed as instructed in the README.md file, the user can train a new model or test a pre-trained one.


## Training a new model
User can choose to train a new model.
During training, the application will show some data about the progression of the training to the user.

Training loss shows the progression of the average loss of a batch and validation accuracy shows the estimated accuracy of the model.

<img src=https://github.com/jooniku/convolutional_neural_network/blob/main/data/training_data_images/25-11-2023-10-38.png height=600 width=550>

## Testing the model
A given model can be tested with different size test-sets (100, 2000, 10 000) of images.

Generally the 2000 image test-set is large enough to provide accurate estimate of the accuracy of the network while being much faster than the complete test of 10 000 images.

After the test the application will give a histogram of correct predictions per class.

## Visualization
User can visualize an image passing through the network.
The app will also give its prediction for the presented image.

<img src= height=400 width=500>

## Week 6

This week was spent writing tests and solving problems with the network. I tested the softmax, loss and gradient functions with hand-computed values. 

I had a mistake where the filters in the convolutional layer used the ouput of the previous filter convolution as input and due to that the network could not learn. I also had an idea of using matrix calculations but decided to not use them since they shouldn't be necessary and would require a large redesign. Currentyl the network learns with one single photo, but using batches or different images doesn't allow the loss to go down. Total time for this week was roughly 25 hours.

Next week I'll solve the batch training issue and will likely have a working network. I'll also write tests for backpropagations. 
This week was mainly spent studying how convolutional neural networks work and how they are implemented. Read roughly 10-15 articles. About 5-7 hours spent reading.

Was able to learn a lot about how CNNs work and how they are done at a higher level. Mainly what are the different layers/processes and why they are done. 

Research is somewhat difficult since most scientific papers are a high-level representation of the network and almost all articles use external libraries to "build" the network. Because of this I really have to spend time fully understanding how and why different things are done the way they are and I can not simply follow pseudo code. 

I was able to create the requirement specification document and start visualizing the network and its processes (e.g. how average pooling algorithm could be implemented). 

However, I am somewhat unsure as to how the neural network itself should be built. I propose a class object (CNN) which would have multiple subclasses such as 3 instances of convolutional layer -classes etc. Is this a reasonable approach? Then images would be given to the CNN class from a file.

I am quite new to CNNs at a lower level, so I think using libraries for things like image preprocessing, mathematics and such would be fine as long as the network and its function themselves are completely constructed "by hand". Is this correct?

Next week I plan to dig deeper in to the last question and work with the different mathematical aspects of the network (such as ReLU implementation). I will write a precise, lower level structure plan of the network and each of its layers in such way that I can then write pseudocode for the different algorithms used. 


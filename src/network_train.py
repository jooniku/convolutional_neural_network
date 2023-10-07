from src.network.neural_network import NeuralNetwork
from src.mnist_data_processor import training_images, training_labels

nn = NeuralNetwork()

nn._train_network()

l = [15, 17, 2, 543, 2212, 88, 44, 2321, 6, 5]
"""
for p in l:
    for i in training_images[p]:
        for j in i:
            if j == 0:
                print("!", end="")
            else:
                print("@", end="")
        print()
"""
amount = 0
for k in l:
    prediction = nn._predict(training_images[k])
    if prediction == training_labels[k]:
        amount += 1

print("Accuracy:", (amount / 10)*100, "%")

from src.network.neural_network import NeuralNetwork
from src.mnist_data_processor import test_images, test_labels
import random
import time
nn = NeuralNetwork()

start_time = time.time()
nn._train_network()
training_time = time.time() - start_time
test_cases = [random.randint(0, 10000) for i in range(1000)]

result = 0
for test in test_cases:
    prediction = nn._predict(test_images[test])
    if prediction == test_labels[test]:
        result += 1

print("Accuracy:", (result / 1000)*100, "%, ", "Training time", training_time, "seconds")

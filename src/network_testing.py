from src.network.neural_network import NeuralNetwork
from src.mnist_data_processor import test_images, test_labels
import random
import time

hyperparameters = []

if False:
    for i in range(15):
        learning_rate = 10 ** random.uniform(-6, 1)
        reg_strength = 10 ** (0.1 * random.uniform(-6, 1))
        nn = NeuralNetwork(learning_step_size=learning_rate, reg_strength=reg_strength, epochs=50)
        
        start_time = time.time()
        nn._train_network()
        training_time = time.time() - start_time
        test_cases = [random.randint(0, 10000-1) for i in range(100)]

        result = 0
        for test in test_cases:
            prediction = nn._predict(test_images[test])
            if prediction == test_labels[test]:
                result += 1

        hyperparameters.append((result, learning_rate, reg_strength))

        print("Accuracy:", (result / 100)*100, "%, ", "Training time", training_time, "seconds")

    print(max(hyperparameters))

if True:
    nn = NeuralNetwork()
    start_time = time.time()
    nn._train_network()
    training_time = time.time() - start_time
    test_cases = [random.randint(0, 10000-1) for i in range(100)]

    result = 0
    for test in test_cases:
        prediction = nn._predict(test_images[test])
        if prediction == test_labels[test]:
            result += 1


    print("Accuracy:", (result / 100)*100, "%, ", "Training time", training_time, "seconds")

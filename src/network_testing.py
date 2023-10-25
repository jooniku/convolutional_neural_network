from src.network.neural_network import NeuralNetwork
from src.mnist_data_processor import test_images, test_labels
import numpy as np

nn = NeuralNetwork()
nn.load_latest_saved_network()


# complete test
if False:
    result = 0
    for test in range(len(test_images)):
        prediction = nn.predict(test_images[test])
        if prediction == test_labels[test]:
            result += 1

    print("Test accuracy:", (result / len(test_images))*100, "%")

# quick test
if True:
    result = 0
    nums = [np.random.randint(0, 10_000) for i in range(500)]

    for num in nums:
        prediction = nn.predict(test_images[num])
        if prediction == test_labels[num]:
            result += 1

    print("Test accuracy:", (result / len(nums))*100, "%")

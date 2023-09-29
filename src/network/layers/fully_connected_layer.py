from layers.softmax import SoftMaxClassifier

class FullyConnectedLayer:
    """_summary_

    Args:
        NeuralNetwork (_type_): _description_
    """

    def __init__(self, learning_step_size, reg_strength) -> None:
        self.softmax = SoftMaxClassifier(learning_step_size=learning_step_size, reg_strength=reg_strength)


    def _process(self, image):
        flattened_image = image.flatten().reshape(1, -1)
        

    def _compute_loss(self, image, kernel, label):
        return self.softmax._compute_loss(image=image, kernel=kernel, label=label)

    def _compute_gradient(self, image, label):
        return self.softmax._compute_gradient(image=image, label=label)
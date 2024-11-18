

class NNConfig:
    def __init__(self, input_size, hidden_size, num_classes, num_epochs, batch_size, learning_rate):
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_classes: int = num_classes
        self.num_epochs: int = num_epochs
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate
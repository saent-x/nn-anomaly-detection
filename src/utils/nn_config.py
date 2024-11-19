

class NNConfig:
    def __init__(self, num_epochs, batch_size, learning_rate, model_path: str, log_level: int):
        self.num_epochs: int = num_epochs
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate

        self.model_path: str = model_path
        self.log_level: int = log_level
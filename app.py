import logging

from can_trainer import CANDataTrainer
from nn_config import NNConfig


def main():
    nn_config = NNConfig(100, 64, 0.001, "can_ad_model.onnx", logging.INFO)

    can_trainer = CANDataTrainer(config=nn_config)
    can_trainer.train("./data/can_dataset/simple.csv")
    
if __name__ == '__main__':
    main()
    
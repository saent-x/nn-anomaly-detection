import torch
import torch.nn as tnn
from torch import Tensor
from torch.utils.data import WeightedRandomSampler, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import math
from dataset import CANDataset
from nn import NeuralNet
import torch.onnx

from nn_config import NNConfig

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model(model, nn_config: NNConfig) -> None:
    # Save the trained model in ONNX format
    model.eval()
    onnx_file_path: str = "can_ad_model.onnx"

    x: Tensor = torch.randn(nn_config.batch_size, nn_config.input_size, requires_grad=True)

    torch.onnx.export(
        model,
        x,
        onnx_file_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['arbitration_id', 'df1', 'df2', 'df3', 'df4', 'df5', 'df6', 'df7', 'df8', 'time_interval'],
        output_names=['attack'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model has been saved in ONNX format at {onnx_file_path}")
    
def run() -> None:
    nn_config = NNConfig(10, 200, 2, 2, 100, 0.001)
    full_dataset = CANDataset()
    
    num_of_samples_in_class_0: int = np.count_nonzero(full_dataset.y == 0)
    num_of_samples_in_class_1: int = np.count_nonzero(full_dataset.y == 1)
        
    train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])
    train_loader = DataLoader(dataset=train_dataset, batch_size=nn_config.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=nn_config.batch_size, shuffle=False, num_workers=2)
    
    model = NeuralNet(nn_config.input_size, nn_config.hidden_size, nn_config.num_classes)
        
    # loss and optimizer
    pos_weight: float = num_of_samples_in_class_0 / num_of_samples_in_class_1 # size of largest class / size of positive class (which is the lowest)
    criterion = tnn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=nn_config.learning_rate)
    
    # training loop
    n_total_steps: int = len(train_loader)
    
    for epoch in range(nn_config.num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % nn_config.batch_size == 0:
                print(f'epoch {epoch+1}/{nn_config.num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
                
    # validation
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        
        for inputs, labels in test_loader:
            outputs = model(inputs)
            
            # value, index
            _, predictions = torch.max(outputs, 1)
            print(predictions)

            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
            
        acc = 100.0 * n_correct / n_samples
        print(f"accuracy = {acc}")
    
    print("Training complete!")

    #save model
    #save_model(model, nn_config)
    
    
if __name__ == '__main__':
    run()
    
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=None, dropout_rate: float = 0.3):
        super(NeuralNet, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32] #TODO can this be same values or changed

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    
        
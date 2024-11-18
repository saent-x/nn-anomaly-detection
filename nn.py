import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        
        self.l1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(0.2)
        self.act2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, 1) # using 1 since its BCEWithLogitsLoss
    
    def forward(self, x) -> None:
        out = self.l1(x)
        out = self.act1(out)
        out = self.l2(out)
        out = self.drop(out)
        out = self.act2(out)
        out = self.l3(out)
        
        return out
    
    
        
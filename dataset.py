import torch
import torchvision
from numpy import ndarray
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math

class CANDataset(Dataset):
    
    def __init__(self, transform=None) -> None:
        xy = np.loadtxt("./data/can_dataset/simple.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples  = xy.shape[0]

        x: ndarray = xy[:, :-1]
        y: ndarray = xy[:, [-1]]

        # apply standard scaler
        scaler = StandardScaler()
        x_scaled: ndarray = scaler.fit_transform(x)
        
        self.x: Tensor = torch.from_numpy(x_scaled)
        self.y: Tensor = torch.from_numpy(y)
        
        self.transform = transform
        
    
    def __getitem__(self, index) -> [Tensor, Tensor]:
        return self.x[index], self.y[index]
        
    def __len__(self) -> None:
        return self.n_samples
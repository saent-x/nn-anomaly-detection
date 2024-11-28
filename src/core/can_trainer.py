import logging
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch import nn, optim
from torch.xpu import device

from src.utils.nn_config import NNConfig
from torch.utils.data import  DataLoader
import numpy as np
from src.core.can_dataset import CANDataset
from src.core.nn import NeuralNet
import torch.onnx
import pickle

class CANDataTrainer:
    def __init__(self, config: NNConfig):
        logging.basicConfig(
            level=config.log_level,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )

        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Training on device: {self.device}\n")


    def _load_and_preprocess_data(self, filepath: str) -> tuple[DataLoader, DataLoader]:
        try:
            df = pd.read_csv(filepath)

            features = ['arbitration_id', 'df1', 'df2', 'df3', 'df4', 'df5', 'df6', 'df7', 'df8', 'time_interval']

            x = df[features].values
            y = df['attack'].values

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42, stratify=y
            )

            self.class_0_count = np.count_nonzero(y_train == 0)
            self.class_1_count = np.count_nonzero(y_train == 1)

            self.scaler = StandardScaler()
            x_train_scaled = self.scaler.fit_transform(x_train)
            x_test_scaled = self.scaler.transform(x_test)

            x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32, device=self.device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device).unsqueeze(1)
            x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32, device=self.device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=self.device).unsqueeze(1)

            train_dataset = CANDataset(x_train_tensor, y_train_tensor)
            test_dataset = CANDataset(x_test_tensor, y_test_tensor)

            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)

            return train_loader, test_loader

        except Exception as e:
            self.logger.error(f"Error loading and normalizing data: {e}")
            raise


    def train(self, filepath: str) -> Optional[nn.Module]:
        try:
            train_loader, test_loader = self._load_and_preprocess_data(filepath)

            model = NeuralNet(input_dim=10).to(self.device)

            pos_weight: float = self.class_0_count / self.class_1_count  # size of largest class / size of positive class (which is the lowest)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=self.device))
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

            for epoch in range(self.config.num_epochs):
                model.train()
                total_loss = 0.0

                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # Validation
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_x)
                        predicted = (outputs >= 0.5).float()
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()

                accuracy = 100 * correct / total

                # Log progress
                self.logger.info(
                    f"Epoch [{epoch + 1}/{self.config.num_epochs}] "
                    f"Loss: {total_loss / len(train_loader):.4f} "
                    f"Accuracy: {accuracy:.2f}%"
                )

            model.eval()
            self._export_to_onnx(model)

            return model

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return None

    def _save_scaler(self) -> None:
        try:
            with open('models/can_dataset_scaler.pkl', 'wb') as f:
                # noinspection PyTypeChecker
                pickle.dump(self.scaler, f)
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")

    def _export_to_onnx(self, model: nn.Module):
        try:
            example_input = torch.randn(1, 10, dtype=torch.float32, device=self.device)

            torch.onnx.export(
                model,
                example_input,
                self.config.model_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['attack_probability'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'attack_probability': {0: 'batch_size'}
                }
            )
            self._save_scaler()

            print()
            self.logger.info(f"Model exported to {self.config.model_path}")
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")




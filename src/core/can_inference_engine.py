import time
import pickle
from datetime import datetime
from tabulate import tabulate

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import onnxruntime as rt
from can import Message

from src.utils.utilities import pad_list


class CanInferenceEngine:
    def __init__(self, model_path: str) -> None:
        self.session = rt.InferenceSession(model_path)
        self.scaler = CanInferenceEngine._load_scaler()
        self.anomaly_count = 0
        self.attack_free_count = 0
        # self.expected_columns = ['arbitration_id', 'df1', 'df2', 'df3', 'df4', 'df5', 'df6', 'df7', 'df8', 'time_interval']

    @staticmethod
    def _load_scaler() -> StandardScaler:
        with open('models/can_dataset_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

            return scaler

    def predict(self, can_message: pd.DataFrame) -> int:
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        input_tensor = can_message.astype(np.float32)

        # Run inference
        predictions = self.session.run([output_name], {input_name: input_tensor})[0]

        binary_predictions = (predictions >= 0.5).astype(int)

        if binary_predictions == 1:
            self.anomaly_count += 1
        elif binary_predictions == 0:
            self.attack_free_count += 1

        return binary_predictions

    def parse_can_message(self, can_message: Message, previous_timestamp) -> tuple[pd.DataFrame, float]:
        arbitration_id = float(can_message.arbitration_id)
        data_fields = [float(byte) for byte in can_message.data[:8]]

        data_fields = pad_list(data_fields)

        current_timestamp = time.time()
        time_interval = 0.0 if previous_timestamp == 0.0 else current_timestamp - previous_timestamp

        message_data = [
            arbitration_id,
            data_fields[0],
            data_fields[1],
            data_fields[2],
            data_fields[3],
            data_fields[4],
            data_fields[5],
            data_fields[6],
            data_fields[7],
            time_interval
        ]
        message_df = pd.DataFrame([message_data])
        message_scaled = self.scaler.transform(message_df)

        return message_scaled, current_timestamp

    def generate_inference_summary(self) -> None:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        heading = f"{current_datetime}: Summary of CAN-AD Session"

        data = [[self.anomaly_count, self.attack_free_count]]
        headers = ["Anomalies", "Attack-Free"]

        print(f"\n\n{heading}")
        print(tabulate(data, headers=headers, tablefmt="grid"))
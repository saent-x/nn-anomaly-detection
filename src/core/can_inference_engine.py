import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import onnxruntime as rt
from can import Message

from src.utils.utilities import pad_list


class CanInferenceEngine:
    def __init__(self, model_path: str) -> None:
        self.session = rt.InferenceSession(model_path)
        self.scaler = StandardScaler()
        self.expected_columns = ['arbitration_id', 'df1', 'df2', 'df3', 'df4', 'df5', 'df6', 'df7', 'df8', 'time_interval']


    def predict(self, can_message: pd.DataFrame) -> int:
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        input_tensor = can_message.astype(np.float32)

        # Run inference
        predictions = self.session.run([output_name], {input_name: input_tensor})[0]

        binary_predictions = (predictions >= 0.5).astype(int)

        return binary_predictions

    def parse_can_message(self, can_message: Message, previous_timestamp) -> tuple[pd.DataFrame, float]:
        arbitration_id = float(can_message.arbitration_id)
        data_fields = [float(byte) for byte in can_message.data[:8]]

        data_fields = pad_list(data_fields)

        current_timestamp = time.time()
        time_interval = 0.0 if previous_timestamp is 0.0 else current_timestamp - previous_timestamp

        message_data = {
            'arbitration_id': arbitration_id,
            'df1': data_fields[0],
            'df2': data_fields[1],
            'df3': data_fields[2],
            'df4': data_fields[3],
            'df5': data_fields[4],
            'df6': data_fields[5],
            'df7': data_fields[6],
            'df8': data_fields[7],
            'time_interval': time_interval
        }

        message_df = pd.DataFrame([message_data], columns=self.expected_columns)
        message_scaled = self.scaler.fit_transform(message_df)

        return message_scaled, current_timestamp

    # TODO: Method to print out summary of inference after an inference session
    def generate_inference_summary(self) -> None:
        pass
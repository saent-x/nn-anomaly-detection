import can
import numpy as np
import pandas as pd
import time
from can import Message
from onnxruntime import InferenceSession
from sklearn.preprocessing import StandardScaler
import onnxruntime as rt


previous_timestamp: float = 0.0

def pad_list(input_list) -> list[float]:
    while len(input_list) < 8:
        input_list.append(0)
    return input_list

def predict(session: InferenceSession, can_message: pd.DataFrame) -> int:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_tensor = can_message.astype(np.float32)

    # Run inference
    predictions = session.run([output_name], {input_name: input_tensor})[0]

    binary_predictions = (predictions >= 0.5).astype(int)

    return binary_predictions


def parse_can_message(can_message: Message) -> dict[str, float]:
    global previous_timestamp

    arbitration_id = float(can_message.arbitration_id)
    data_fields = [float(byte) for byte in can_message.data[:8]]

    data_fields = pad_list(data_fields)

    current_timestamp = time.time()

    time_interval = 0.0 if previous_timestamp is 0.0 else current_timestamp - previous_timestamp
    previous_timestamp = current_timestamp

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

    return message_data

def process_can_message(can_message: Message, session: InferenceSession, scaler: StandardScaler) -> None:
    expected_columns = ['arbitration_id', 'df1', 'df2', 'df3', 'df4', 'df5', 'df6', 'df7', 'df8', 'time_interval']

    message_data = parse_can_message(can_message)
    message_df = pd.DataFrame([message_data], columns=expected_columns)
    message_scaled = scaler.fit_transform(message_df)

    prediction = predict(session, message_scaled)

    if prediction == 1:
        print("Prediction (Binary): Anomaly Detected\n")
    elif prediction == 0:
        print("Prediction (Binary): No Anomaly Detected\n")


def main():
    can_interface = 'vcan0'
    bus = can.interface.Bus(can_interface, interface='socketcan')

    session = rt.InferenceSession("can_an_model.onnx")

    try:
        scaler = StandardScaler()
        print("Listening on CAN interface...\n")

        while True:
            message: Message = bus.recv(timeout=1.0)

            if message:
                print(f"Received CAN message: {message.arbitration_id} {message.data}")
                process_can_message(message, session, scaler)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        bus.shutdown()



if __name__ == '__main__':
    main()

import can
from can import Message

from src.core.can_inference_engine import CanInferenceEngine


previous_timestamp: float = 0.0

def process_can_message(can_message: Message, can_inf_eng: CanInferenceEngine) -> None:
    global previous_timestamp

    message_data, current_timestamp = can_inf_eng.parse_can_message(can_message, previous_timestamp)
    previous_timestamp = current_timestamp  # update timestamp here since the variable is tracked here

    prediction = can_inf_eng.predict(message_data)

    if prediction == 1:
        print(f"Prediction (Binary): Anomaly Detected\n")
    elif prediction == 0:
        print(f"Prediction (Binary): Attack Free\n")


def main() -> None:
    can_interface = 'vcan0'

    bus = can.interface.Bus(can_interface, interface='socketcan')
    can_inf_eng = CanInferenceEngine("models/can_ad_full_model.onnx")

    try:
        print("Listening on CAN interface...\n")

        while True:
            message: Message = bus.recv(timeout=1.0)

            if message:
                print(f"Received CAN message: ARBITRATION ID: {message.arbitration_id} DATA-FIELD: [{message.data.hex(sep=' ')}]")
                process_can_message(message, can_inf_eng)

    except KeyboardInterrupt:
        print("Stopped abruptly by user.")
    finally:
        # print summary
        can_inf_eng.generate_inference_summary()

        bus.shutdown()



if __name__ == '__main__':
    main()

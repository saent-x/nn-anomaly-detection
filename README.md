# CAN Bus Anomaly Detection

A PyTorch neural network that flags malicious frames on a vehicle CAN bus, exported to ONNX and served by a lightweight inference loop that listens on a live SocketCAN interface.

Part of my in-vehicle intrusion detection work. See also [Automotive-IDS-ML](https://github.com/saent-x/Automotive-IDS-ML) (hybrid signature + anomaly IDS) and [neural-network-module](https://github.com/saent-x/neural-network-module) (the same idea with a neural network written from scratch in Go).

## How it works

```
raw CAN logs ──► preprocessor ──► features ──► PyTorch MLP ──► ONNX model
                                                                  │
live CAN bus (vcan0) ──► parse frame ──► scale ──► onnxruntime ──► "Attack Free" / "Anomaly Detected"
```

**Features (10 per frame):** arbitration ID, the 8 data bytes (zero-padded), and the time since the previous frame.

**Model:** a fully connected network (`10 → 64 → 32 → 1`) with BatchNorm, ReLU and dropout, trained with `BCEWithLogitsLoss` and a positive-class weight to handle the attack/normal class imbalance. Data is split 80/20 with stratification and standardised with a `StandardScaler`, which is saved next to the model so inference uses the same scaling.

**Inference:** `inference.py` reads frames from a SocketCAN interface with `python-can`, builds the same feature vector, and runs the exported ONNX model with `onnxruntime`. No PyTorch is needed at inference time.

## Layout

| Path | What |
| --- | --- |
| `scripts/can_dataset_preprocessor.py` | Merges labelled CAN capture CSVs into one training file |
| `src/core/nn.py` | Network definition |
| `src/core/can_trainer.py` | Training loop, evaluation and ONNX export |
| `src/core/can_inference_engine.py` | Frame parsing, scaling and ONNX inference |
| `app.py` | Trains a model from `data/can_dataset/can.csv` |
| `inference.py` | Live detection loop on `vcan0` |
| `Dockerfile` | Slim inference-only image |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train (expects data/can_dataset/can.csv)
python app.py

# live detection on a virtual CAN interface (Linux)
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan && sudo ip link set up vcan0
python inference.py
```

Or run inference in Docker (the container needs access to the host's CAN interface):

```bash
docker build -t can-ad .
docker run --network host can-ad
```

## Stack

Python · PyTorch · scikit-learn · ONNX / onnxruntime · python-can · Docker

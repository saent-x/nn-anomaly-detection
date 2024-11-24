FROM python:3.10

COPY inference.py .

COPY models/can_ad_model.onnx models/can_ad_model.onnx
COPY models/can_dataset_scaler.pkl models/can_dataset_scaler.pkl
COPY src src

RUN pip install tabulate scikit-learn numpy pandas python-can onnx onnxruntime

CMD ["python", "./inference.py"]


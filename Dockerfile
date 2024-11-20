FROM python:3.10

ADD inference.py .

ADD nn-ad-torch-venv/can_classifier.keras .

RUN pip install python-can pandas keras scikit-learn tensorflow

CMD ["python", "./can_anomaly_detection.py"]


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def load_data_in_chunks(file_path, chunk_size=10000):
    """
    Generator that reads data in chunks and yields preprocessed data batches.
    """
    scaler = StandardScaler()

    # Initialize scaler with the first chunk to fit the scale
    initial_chunk = pd.read_csv(file_path, chunksize=chunk_size).__next__()
    X_initial = initial_chunk.drop('attack', axis=1)
    y_initial = initial_chunk['attack']
    scaler.fit(X_initial)  # Fit scaler on first chunk only

    def data_generator():
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            X = chunk.drop('attack', axis=1)
            y = chunk['attack']
            X = scaler.transform(X)  # Transform each chunk separately
            yield X, y

    return data_generator()

def build_model(input_shape):
    """
    Build the neural network model.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC'])

    return model

def train_model(model, data_generator, steps_per_epoch, epochs=10):
    """
    Train the model using data in chunks.
    """
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]

    history = model.fit(
        data_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks
    )

    return history

def main():
    # Load and preprocess data generator
    file_path = 'data/simple.csv'
    chunk_size = 10000  # Adjust chunk size based on your memory capacity

    data_gen = load_data_in_chunks(file_path, chunk_size)
    # Assuming each chunk represents one step per epoch
    steps_per_epoch = sum(1 for _ in pd.read_csv(file_path, chunksize=chunk_size))

    # Build and train the model
    input_shape = next(data_gen)[0].shape[1:]
    model = build_model(input_shape)
    train_model(model, data_gen, steps_per_epoch)

    # Save the model
    model.save('can_classifier.keras')

if __name__ == '__main__':
    main()

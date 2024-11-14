import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def load_and_preprocess_data(file_path):
    """
    Load, balance (upsample), and preprocess the CAN dataset.
    """
    df = pd.read_csv(file_path)

    # Drop rows with missing values
    df = df.dropna()

    # Check class distribution in the 'attack' column
    class_counts = df['attack'].value_counts()
    print("Class distribution before balancing:")
    print(class_counts)

    # Balance the dataset by upsampling the minority class
    if class_counts.min() != class_counts.max():
        # Separate majority and minority classes
        df_majority = df[df['attack'] == class_counts.idxmax()]
        df_minority = df[df['attack'] == class_counts.idxmin()]

        # Upsample the minority class
        df_minority_upsampled = df_minority.sample(class_counts.max(), replace=True, random_state=42)
        df = pd.concat([df_majority, df_minority_upsampled], ignore_index=True)

    # Verify new class distribution
    print("Class distribution after balancing:")
    print(df['attack'].value_counts())

    # Split features and target
    X = df.drop('attack', axis=1)
    y = df['attack']

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val



def build_model(_input_shape):
    """
    Build the neural network model.
    """
    model = keras.Sequential([
        layers.Input(shape=_input_shape),
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

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the model and evaluate its performance.
    """
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]

    history = model.fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=100,
                       batch_size=32,
                       callbacks=callbacks)

    _, accuracy, auc = model.evaluate(X_val, y_val)
    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f'Validation AUC: {auc:.4f}')

    y_pred = model.predict(X_val)
    y_pred_binary = (y_pred > 0.5).astype(int)
    print('Classification Report:')
    print(classification_report(y_val, y_pred_binary))

    return history

def predict():
    df = pd.read_csv("data/inference.csv")

    # Drop rows with missing values
    df = df.dropna()

    # Split features and target
    X_inference = df

    loaded_model = keras.models.load_model("can_classifier.keras")
    loaded_model.summary()

    predictions = loaded_model.predict(X_inference)
    predictions_binary = (predictions > 0.5).astype(int)

    print(predictions_binary)


def main():
    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_and_preprocess_data('data/simple.csv')

    # Build and train the model
    # _input_shape = X_train.shape[1:]
    # model = build_model(_input_shape)
    # train_model(model, X_train, y_train, X_val, y_val)
    #
    # # Save the model
    # model.save('can_classifier.keras')

if __name__ == '__main__':
    #predict()
    main()

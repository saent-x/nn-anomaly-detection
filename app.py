import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import pandas as pd
import keras
import tensorflow as tf
import datetime

from metrics import plot_training_history, evaluate_model_performance
from keras import layers


def convert_dataframe_to_dataset(df):
    df = df.copy()

    labels = df.pop('attack')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=10000)

    return ds

def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = layers.Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature

def prepare_train_and_test_ds(filepath):
    df = pd.read_csv(filepath)

    validation_df = df.sample(frac=0.2, random_state=1337)
    training_df = df.drop(validation_df.index)

    print(
        f"Using {len(training_df)} samples for training "
        f"and {len(validation_df)} for validation"
    )

    train_ds = convert_dataframe_to_dataset(training_df)
    val_ds = convert_dataframe_to_dataset(validation_df)

    all_inputs = {
        'arbitration_id': keras.Input(shape=(1,), name='arbitration_id'),
        'df1': keras.Input(shape=(1,), name='df1'),
        'df2': keras.Input(shape=(1,), name='df2'),
        'df3': keras.Input(shape=(1,), name='df3'),
        'df4': keras.Input(shape=(1,), name='df4'),
        'df5': keras.Input(shape=(1,), name='df5'),
        'df6': keras.Input(shape=(1,), name='df6'),
        'df7': keras.Input(shape=(1,), name='df7'),
        'df8': keras.Input(shape=(1,), name='df8'),
        'time_interval': keras.Input(shape=(1,), name='time_interval')
    }

    encoded_features = []
    for name, input_tensor in all_inputs.items():
        encoded = encode_numerical_feature(input_tensor, name, train_ds)
        encoded_features.append(encoded)

    all_features = layers.concatenate(encoded_features)

    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)

    return train_ds, val_ds, all_features, all_inputs


def train_model():
    train_ds, val_ds, all_features, all_inputs = prepare_train_and_test_ds("data/full_processed_can_data.csv")

    x = layers.Dense(32, activation="relu")(all_features)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs = all_inputs, outputs = output)
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    # Cache the datasets for better performance
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

    # Add prefetch to optimize data pipeline
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    history = model.fit(train_ds, epochs=30, validation_data=val_ds,
                        callbacks=[
                            keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                patience=5,
                                restore_best_weights=True
                            ),
                            keras.callbacks.ReduceLROnPlateau(
                                monitor='val_loss',
                                factor=0.2,
                                patience=3
                            )
                        ])

    # Plot training history
    training_metrics = plot_training_history(history)

    # Evaluate model
    evaluation_metrics = evaluate_model_performance(model, val_ds)

    # Save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model.save(f"model_{timestamp}.keras")

    return model, training_metrics, evaluation_metrics


def main():
    model, training_metrics, evaluation_metrics = train_model()

    print("\nTraining Metrics:")
    for metric, value in training_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nEvaluation Metrics:")
    print(f"ROC AUC: {evaluation_metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(evaluation_metrics['classification_report'])




if __name__ == '__main__':
    main()


import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import pandas as pd
import keras
import tensorflow as tf
import datetime
import numpy as np

from metrics import plot_training_history, evaluate_model_performance
from keras import layers
# from keras.src.saving import serialization_lib
#
# serialization_lib.enable_unsafe_deserialization()


def convert_dataframe_to_dataset(df):
    df = df.copy()

    labels = df.pop('attack')
    labels = labels.values.reshape(-1, 1) # reshape to match model output dims
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=10000)

    return ds


def encode_numerical_feature(feature, name, dataset, tf):
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_values = np.array(list(feature_ds.as_numpy_iterator()))

    zero_fraction = (feature_values == 0).mean()
    value_range = feature_values.max() - feature_values.min()

    if name == 'time_interval':
        normalizer = layers.Normalization(axis=None)
        feature_ds = dataset.map(lambda x, y: x[name])
        feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
        normalizer.adapt(feature_ds)
        encoded_feature = tf.keras.layers.Lambda(
            lambda x: normalizer(tf.expand_dims(x, -1)),
            output_shape=(1,)
        )(feature)

    elif zero_fraction > 0.3:
        normalizer = layers.Normalization(axis=None)
        feature_ds = dataset.map(lambda x, y: x[name])
        feature_ds = feature_ds.map(lambda x: tf.expand_dims(tf.cast(x, tf.float32) + 1e-6, -1))
        normalizer.adapt(feature_ds)
        encoded_feature = tf.keras.layers.Lambda(
            lambda x: normalizer(tf.expand_dims(tf.cast(x, tf.float32) + 1e-6, -1)),
            output_shape=(1,)
        )(feature)

    elif value_range > 1000:
        normalizer = layers.Normalization(axis=None)
        feature_ds = dataset.map(lambda x, y: x[name])
        feature_ds = feature_ds.map(lambda x: tf.expand_dims(tf.math.log1p(tf.cast(x, tf.float32)), -1))
        normalizer.adapt(feature_ds)
        encoded_feature = tf.keras.layers.Lambda(
            lambda x: normalizer(tf.expand_dims(tf.math.log1p(tf.cast(x, tf.float32)), -1)),
            output_shape=(1,)
        )(feature)

    else:
        normalizer = layers.Normalization(axis=None)
        feature_ds = dataset.map(lambda x, y: x[name])
        feature_ds = feature_ds.map(lambda x: tf.expand_dims(tf.cast(x, tf.float32), -1))
        normalizer.adapt(feature_ds)
        encoded_feature = tf.keras.layers.Lambda(
            lambda x: normalizer(tf.expand_dims(tf.cast(x[name], tf.float32), -1)),
            output_shape=(1,)
        )(feature)

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

    print("\ndone converting to DS...")

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

    print("loaded all inputs...")

    encoded_features = []
    for name, input_tensor in all_inputs.items():
        encoded = encode_numerical_feature(input_tensor, name, train_ds, tf)
        encoded_features.append(encoded)

    all_features = layers.concatenate(encoded_features)

    print("done encoding and concatenating...")

    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)

    return train_ds, val_ds, all_features, all_inputs


def train_model():
    train_ds, val_ds, all_features, all_inputs = prepare_train_and_test_ds("data/simple.csv") #full_processed_can_data

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

    print("started model training...")
    

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

    training_metrics = plot_training_history(history)
    evaluation_metrics = evaluate_model_performance(model, val_ds)
    #timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model.save(f"new_model_20241111-123230.keras")

    print("done model training...")

    return model, training_metrics, evaluation_metrics

def create_can_sample(arbitration_id, df1, df2, df3, df4, df5, df6, df7, df8, time_interval, tf):
    return [arbitration_id, df1, df2, df3, df4, df5, df6, df7, df8, time_interval]

def predict():

    sample = create_can_sample(170, 0, 0, 0, 0, 0, 0, 0, 0, 0.00048089027404785156, tf)
    loaded_model = keras.models.load_model("new_model_20241111-123230.keras")
    loaded_model.summary()

    input_tensor = tf.convert_to_tensor([sample])
    input_tensors = tf.split(input_tensor, 10, axis=1)

    input_dict = {
        'arbitration_id': input_tensors[0],
        'df1': input_tensors[1],
        'df2': input_tensors[2],
        'df3': input_tensors[3],
        'df4': input_tensors[4],
        'df5': input_tensors[5],
        'df6': input_tensors[6],
        'df7': input_tensors[7],
        'df8': input_tensors[8],
        'time_interval': input_tensors[9]
    }

    predictions = loaded_model.predict(input_dict)

    print(f"prediction of sample: {100 * predictions[0][0]:.1f}")

def create_model():
    model, training_metrics, evaluation_metrics = train_model()

    print("\nTraining Metrics:")
    for metric, value in training_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nEvaluation Metrics:")
    print(f"ROC AUC: {evaluation_metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(evaluation_metrics['classification_report'])

def main():
    create_model()
    #predict()




if __name__ == '__main__':
    main()


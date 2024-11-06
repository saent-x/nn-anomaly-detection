from xml.sax.handler import all_features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

import keras
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras import layers
from keras.api.utils import to_categorical

import tensorflow as tf


def ConvertDataframeToDataset(df):
    df = df.copy()

    labels = df.pop('attack')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=10000)

    return ds

def EncodeNumericalFeature(feature, name, dataset):
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

def PrepareTrainAndTestDS(filepath):
    df = pd.read_csv(filepath)

    validation_df = df.sample(frac=0.2, random_state=1337)
    training_df = df.drop(validation_df.index)

    print(
        f"Using {len(training_df)} samples for training "
        f"and {len(validation_df)} for validation"
    )

    train_ds = ConvertDataframeToDataset(training_df)
    val_ds = ConvertDataframeToDataset(validation_df)

    for x, y in train_ds.take(1):
        print("Input:", x)
        print("Target:", y)

    # batch datasets
    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)

    # feature processing for numerical features
    arbitration_id = keras.Input(shape=(1,), name='arbitration_id')
    df1 = keras.Input(shape=(1,), name='df1')
    df2 = keras.Input(shape=(1,), name='df2')
    df3 = keras.Input(shape=(1,), name='df3')
    df4 = keras.Input(shape=(1,), name='df4')
    df5 = keras.Input(shape=(1,), name='df5')
    df6 = keras.Input(shape=(1,), name='df6')
    df7 = keras.Input(shape=(1,), name='df7')
    df8 = keras.Input(shape=(1,), name='df8')
    time_interval = keras.Input(shape=(1,), name='time_interval'),

    arbitration_id_norm = EncodeNumericalFeature(arbitration_id, "arbitration_id", train_ds)
    df1_norm = EncodeNumericalFeature(df1, "df1", train_ds)
    df2_norm = EncodeNumericalFeature(df2, "df2", train_ds)
    df3_norm = EncodeNumericalFeature(df3, "df3", train_ds)
    df4_norm = EncodeNumericalFeature(df4, "df4", train_ds)
    df5_norm = EncodeNumericalFeature(df5, "df5", train_ds)
    df6_norm = EncodeNumericalFeature(df6, "df6", train_ds)
    df7_norm = EncodeNumericalFeature(df7, "df7", train_ds)
    df8_norm = EncodeNumericalFeature(df8, "df8", train_ds)
    time_interval_norm = EncodeNumericalFeature(time_interval, "time_interval", train_ds)

    features = layers.concatenate([
        arbitration_id_norm,
        df1_norm, df2_norm, df3_norm, df4_norm, df5_norm, df6_norm, df7_norm, df8_norm,
        time_interval_norm
    ])

    return train_ds, val_ds, features


def TrainModel():
    train_ds, val_ds, features = PrepareTrainAndTestDS("data/full_processed_can_data.csv")

    x = layers.Dense(32, activation="relu")(features)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs = features, outputs = output)
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    model.fit(train_ds, epochs=30, validation_data=val_ds) #TODO increase epochs count

    # save model
    model.save("model.keras")


def main():
    TrainModel()




if __name__ == '__main__':
    main()


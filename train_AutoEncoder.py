# Copyright 2023 Andreas Koehler, MIT license

"""
Code for training an auto-encoder model 

"""

import numpy as np
import tensorflow as tf
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tensorflow.keras import Sequential

class CAE(tf.keras.Model):
    def __init__(self, input_shape, latent_factor = 2, base_depth=64):
        super(CAE, self).__init__()

        init_conv1d = lambda d, f, s, a: tf.keras.Sequential([tf.keras.layers.Conv1D(d,
                                                            f,
                                                            strides=s,
                                                            padding='same',
                                                            kernel_initializer='glorot_normal'),
                                                            tf.keras.layers.BatchNormalization(),
                                                            tf.keras.layers.Activation(a)])

        init_conv1dtranspose = lambda d, f, s, a: tf.keras.Sequential([tf.keras.layers.Conv1DTranspose(d,
                                                             f,
                                                             strides=s,
                                                             padding='same',
                                                             kernel_initializer='glorot_normal'),
                                                                    tf.keras.layers.BatchNormalization(),
                                                                    tf.keras.layers.Activation(a)])

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            init_conv1d(base_depth, 7, 2, 'elu'),
            init_conv1d(2 * base_depth, 7, 2, 'elu'),
            init_conv1d(2 * base_depth, 7, 2, 'elu'),
            init_conv1d(4 * base_depth, 7, 2, 'elu'),
            init_conv1d(4 * base_depth, 7, 2, 'elu'),
            init_conv1d(int(latent_factor * base_depth), 7, 2, 'linear')
        ])

        output_shape = self.encoder.layers[-1].output_shape[1:]

        self.encoder.summary()

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=output_shape),
            #tf.keras.layers.Reshape(output_shape),
            init_conv1dtranspose(int(latent_factor * base_depth), 7, 2, 'elu'),
            init_conv1dtranspose(4 * base_depth, 7, 2, 'elu'),
            init_conv1dtranspose(4 * base_depth, 7, 2, 'elu'),
            init_conv1dtranspose(2 * base_depth, 7, 2, 'elu'),
            init_conv1dtranspose(2 * base_depth, 7, 2, 'elu'),
            init_conv1dtranspose(base_depth, 7, 2, 'elu'),
        ])
        self.decoder.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_shape[-1])))
        self.decoder.summary()

    def call(self, inputs, training=False):
        return self.decoder(self.encoder(inputs, training=training), training=training)

def main(station,latent_factor=2):
    BATCH_SIZE = 32
    try:
        train = np.load(f'tf/data/train_{station}.npy')
    except FileNotFoundError as e:
        print(e)
        return False

    Xtrain = train
    Xtrain = TimeSeriesScalerMinMax((-1, 1)).fit_transform(Xtrain)

    epochs = 200
    model = CAE(Xtrain.shape[1:], latent_factor = latent_factor, base_depth=64)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer, 'mse', metrics=['mae'])

    model.fit(Xtrain,
              Xtrain,
              epochs=epochs,
              batch_size=BATCH_SIZE,
              callbacks=[tf.keras.callbacks.EarlyStopping('loss', patience=27, restore_best_weights=False),
                         tf.keras.callbacks.ReduceLROnPlateau('loss', patience=11),
                         tf.keras.callbacks.TerminateOnNaN()]
              )

    model.save_weights(f'tf/output/{station}_bn_elu.h5', save_format='h5')
    p_train = model.predict(Xtrain, batch_size=BATCH_SIZE)
    np.save(f'tf/output/{station}_reconstructed_train.npy', p_train)
    np.save(f'tf/output/{station}_reconstruction_error_train.npy', np.mean(abs(p_train-Xtrain),axis=(1,2)))
    p_train = model.encoder.predict(Xtrain, batch_size=BATCH_SIZE)
    np.save(f'tf/output/{station}_encoded_train.npy', p_train)

    return True

if __name__ == '__main__':
    station = 'OSLN2'
    #station = 'EKBG1'
    # number of latent features = latent_factor x time_series_size (length of one component)
    latent_factor = 2
    success = main(station,latent_factor)
    print('Success:', success)

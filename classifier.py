# Copyright 2023 Andreas Koehler, MIT license

import tensorflow as tf
import tensorflow.keras.layers as tfl
from sklearn.model_selection import KFold
import keras_tuner as kt
import numpy as np

class RandomCrop1D(tf.keras.layers.Layer):

    def __init__(self, crop=0.1, name='RandomCrop1D'):
        """
        Crop waveform data randomly.

        Args:
            crop (float) : proportion to crop. default 0.1.
            name (str) : Defaults to RandomCrop1D
        """
        super(RandomCrop1D, self).__init__(name=name)
        self.crop = crop

    def get_config(self):
        return dict(crop=self.crop, name=self.name)

    def build(self, input_dim):
        _, x_size, y_size = input_dim
        self.length = int(x_size * (1 - self.crop))
        self.channels = y_size
        self.rc_layer = tf.keras.layers.RandomCrop(self.length, self.channels)

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)
        x = self.rc_layer(x)
        x = tf.squeeze(x, axis=-1)
        return x

class AlexNet1D(tf.keras.Model):
    def __init__(self, 
                 kernel_sizes=None, 
                 filters=None, 
                 num_outputs=None, 
                 output_type='binary', 
                 pooling='max',
                 name='AlexNet1D'):
        """1D AlexNet

        Args:
            kernel_sizes (list, optional): list of kernel sizes. Defaults to None.
            filters (list, optional): list of number of filters. Defaults to None.
            num_outputs (int, optional): number of outputs. Defaults to None.
            output_type (str, optional): problem type, 'multiclass', 'multilabel'. Defaults to 'binary'.
            pooling (str, optional): pooling type. Defaults to 'max'.
            name (str, optional): model name. Defaults to 'AlexNet1D'.
        """
        super(AlexNet1D, self).__init__(name=name)
        if kernel_sizes is None:
            kernel_sizes = [11, 5, 3, 3, 3]
        if filters is None:
            filters = [96, 256, 384, 384, 256]
        
        assert len(kernel_sizes) == 5
        assert pooling in [None, 'none', 'max', 'avg']

        if pooling == 'max':
            pooling_layer = tfl.MaxPooling1D
        elif pooling == 'avg':
            pooling_layer = tfl.AveragePooling1D
        else:
            pooling_layer = lambda **kwargs: tfl.Activation('linear')

        self.ls = [
            tfl.Conv1D(filters=filters[0], kernel_size=kernel_sizes[0], strides=4, activation='relu',
                                   padding='same'),
            tfl.BatchNormalization(),
            pooling_layer(pool_size=3, strides=2),
            tfl.Conv1D(filters=filters[1], kernel_size=kernel_sizes[1], strides=1, activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            pooling_layer(pool_size=3, strides=2),
            tfl.Conv1D(filters=filters[2], kernel_size=kernel_sizes[2], strides=1, activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv1D(filters=filters[3], kernel_size=kernel_sizes[3], strides=1, activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            tfl.Conv1D(filters=filters[4], kernel_size=kernel_sizes[4], strides=1, activation='relu',
                                   padding="same"),
            tfl.BatchNormalization(),
            pooling_layer(pool_size=3, strides=2),
            tfl.Flatten(),
            tfl.Dense(4096, activation='relu'),
            tfl.Dropout(0.5),
            tfl.Dense(4096, activation='relu'),
            tfl.Dropout(0.5),
        ]

        if num_outputs is not None:
            if output_type == 'binary':
                assert num_outputs == 1
                act = 'sigmoid'
            elif output_type == 'multiclass':
                assert num_outputs > 1
                act = 'softmax'
            elif output_type == 'multilabel':
                assert num_outputs > 1
                act = 'sigmoid'
            else:
                act = 'linear'

            self.ls.append(tfl.Dense(num_outputs, activation=act))

    def call(self, inputs):
        x = inputs
        for layer in self.ls:
            x = layer(x)
        return x

class CVTuner(kt.engine.tuner.Tuner):
    """
    Add cross validation to keras tuner.
    """

    def run_trial(self, trial, x, y, batch_size=32, epochs=1, cv=KFold(5), callbacks=None, **kwargs):
        """
        batch_size : int
        epochs : int
        cv : cross validation splitter.
            Should have split method that accepts x and y and returns train and test indicies.
        callbacks : function that returns keras.callbacks.Callback instaces (in a list).
            eg. callbacks = lambda : [keras.Callbacks.EarlyStopping('val_loss')]
        """
        val_metrics = []
        oof_p = np.zeros(y.shape)
        for train_indices, test_indices in cv.split(x, y):
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            model = self.hypermodel.build(trial.hyperparameters)

            if callbacks is not None:
                cb = callbacks()
            else:
                cb = None

            model.fit(x_train, y_train,
                      validation_data=(x_test, y_test),
                      batch_size=batch_size,
                      epochs=epochs,
                      callbacks=cb,
                      **kwargs)
            val_metrics.append(model.evaluate(x_test, y_test))
            metrics = model.metrics_names

        val_metrics = np.mean(np.asarray(val_metrics), axis=0)
        res = dict(zip(metrics,val_metrics))

        return res


class AlexNetAndreas(tf.keras.Model):
    def __init__(self,
                 kernel_sizes=None,
                 filters=None,
                 num_outputs=None,
                 output_type='binary',
                 pooling='max',
                 name='AlexNet1D'):
        super(AlexNetAndreas, self).__init__(name=name)
        self.backbone = AlexNet1D(kernel_sizes=kernel_sizes,
                                  filters=filters,
                                  num_outputs=num_outputs,
                                  output_type=output_type,
                                  pooling=pooling,
                                  name=name)

    def call(self, inputs, training=False):
        x = inputs
        x = self.backbone(x)
        return x

def print_layer_output_shapes(model_instance, input_shape):
    # Create a dummy input tensor with the correct input shape
    x = tf.keras.Input(shape=input_shape)

    # Initialize a variable to hold the output of each layer
    y = x

    # Pass the dummy input through the model's layers to determine output shapes
    for layer in model_instance.backbone.ls:
        y = layer(y)
        print(f"{layer.name}: {y.shape}")


# Copyright 2023 Andreas Koehler, MIT license

"""
Code for training an blast classifier model 

"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import keras_tuner as kt
import pickle

from classifier import RandomCrop1D, AlexNet1D, CVTuner, AlexNetAndreas, print_layer_output_shapes

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def create_model(hp, num_outputs, crop=False, input_dim = None):
    model = tf.keras.Sequential()
    if crop:
        model.add(RandomCrop1D(hp.Float('random_crop',0.0,0.5,step=0.1,default=0.0)))
    # Filter length (output dimension time x) - is being tuned
    # if best hyper paramerters hp are given this is not used
    d = [29,11,7,7,7]
    ks = [hp.Int(f'kernel_size_{i}',d[i]//2,49,default=d[i]) for i in range(len(d))]
    # number of filters (output dimension y) not tuned
    filters = [96, 256, 384, 384, 256]
    model.add(AlexNet1D(kernel_sizes=ks,
                          filters=filters,
                          num_outputs=num_outputs,
                          output_type='multiclass',
                          pooling=hp.Choice('pooling', ['max','avg'])))

    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(tf.keras.optimizers.Adam(1e-4),
                  loss=loss,
                  metrics=['accuracy'])

    # Dummy model just to print the correct output shapes
    modeldummy = AlexNetAndreas(kernel_sizes=[hp.values[f'kernel_size_{i}'] for i in range(len(d))],
                       filters=filters, num_outputs=num_outputs, output_type='multiclass',
                       pooling=hp.values['pooling'])
    print_layer_output_shapes(modeldummy, input_shape=(int((1.0-hp.values['random_crop'])*input_dim), 3))
    return model


if __name__ == '__main__':

    #SETTINGS
    folds = 5
    crop = True
    tune = False
    batchsize = 32
    rootpath ='./'

    print('Loading data from',rootpath+'tf/data')
    X = np.load(f'{rootpath}/tf/data/blastclassifier_data_crop_{crop}.npy')
    y = np.load(f'{rootpath}/tf/data/blastclassifier_labels_crop_{crop}.npy')
    events = np.load(f'{rootpath}/tf/data/blastclassifier_timewindows_crop_{crop}.npy', allow_pickle=True)
    classes = ['blast' 'noise']

    print('Data size = ', X.shape)
    print('Lable size = ', y.shape)

    X = TimeSeriesScalerMeanVariance().fit_transform(X)

    if tune:
        tuner = CVTuner(
            hypermodel=lambda hp: create_model(hp,num_outputs=len(np.unique(y)),crop=crop),
            oracle=kt.oracles.BayesianOptimizationOracle(
                objective='accuracy',
                max_trials=20,
                num_initial_points=3
            ),
            directory=rootpath+'tf/output',
            project_name=f'blast_wave_cnn_crop_{crop}',
            overwrite=True)

        tuner.search(X[:len(y)], y,
                     cv=StratifiedKFold(folds),
                     epochs=100,
                     batch_size=batchsize,
                     class_weight=dict(zip(np.unique(y), compute_class_weight('balanced', classes=np.unique(y), y=y))),
                     callbacks=lambda : [tf.keras.callbacks.EarlyStopping('val_accuracy', patience=3)])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        tuner.results_summary()
        with open(f'{rootpath}tf/output/best_hps_crop_{crop}.pkl', 'wb') as f:
            pickle.dump(best_hps, f, pickle.HIGHEST_PROTOCOL)

        with open(f'{rootpath}tf/output/best_hps_crop_{crop}.pkl', 'rb') as f:
            best_hps = pickle.load(f)
    else:
        try:
            with open(f'{rootpath}tf/output/best_hps_crop_{crop}.pkl', 'rb') as f:
                best_hps = pickle.load(f)

        except FileNotFoundError:
            print("Hyper parameter file not found")
            exit()

    # final training with best hyperparameters
    hps = best_hps
    oof = np.zeros((y.shape[0], np.unique(y).shape[0]))
    for i, (train_idx, test_idx) in enumerate(StratifiedKFold(folds, shuffle=True).split(X[:len(y)], y)):
        model = create_model(hps, len(np.unique(y)),crop=crop, input_dim=len(X[0]))

        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]

        model.fit(Xtr, ytr,
                  validation_data=(Xte, yte),
                  epochs=100,
                  batch_size=batchsize,
                  sample_weight=compute_sample_weight('balanced', y=ytr),
                  callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy', patience=7),
                          tf.keras.callbacks.ReduceLROnPlateau('val_accuracy', patience=3, factor=0.5, verbose=1)])

        p = model.predict(Xte)
        oof[test_idx] += p

        model.save_weights(f'{rootpath}tf/output/models/blast_fold_{i}_weights_crop_{crop}.h5')

        del model
        tf.keras.backend.clear_session()

    oof = np.argmax(oof, axis=1)

    with open(f'{rootpath}tf/output/blast_oof_results_crop_{crop}.txt','w') as f:
        f.write(','.join(classes) + '\n')
        f.write(f'Precision {precision_score(y, oof, average=None)} \n')
        f.write(f'Recall {recall_score(y, oof, average=None)} \n')
        f.write(f'F1 {f1_score(y, oof, average=None)} \n')
        f.write(f'Balanced Accuracy {balanced_accuracy_score(y, oof)} \n')

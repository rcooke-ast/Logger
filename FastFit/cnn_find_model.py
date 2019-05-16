import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import pdb
import numpy as np
from utilities import load_atomic
from numpy import mean
from numpy import std
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from matplotlib import pyplot as plt

nHIwav = 15
atmdata = load_atomic(return_HIwav=False)
ww = np.where(atmdata["Ion"] == "1H_I")
HIwav = atmdata["Wavelength"][ww][3:]
HIfvl = atmdata["fvalue"][ww][3:]


def load_dataset(zem=3.0, snr=0, ftrain=0.75, numspec=20):
    zstr = "zem{0:.2f}".format(zem)
    sstr = "snr{0:d}".format(int(snr))
    extstr = "{0:s}_{1:s}_nspec{2:d}".format(zstr, sstr, numspec)
    fdata_all = np.load("train_data/cnn_qsospec_fluxspec_{0:s}_fluxonly.npy".format(extstr))
    label_all = np.load("train_data/cnn_qsospec_fluxspec_{0:s}_labelonly.npy".format(extstr))
    ntrain = int(ftrain*fdata_all.shape[0])
    trainX = fdata_all[:ntrain, :]
    trainy = label_all[:ntrain, :]
    testX = fdata_all[ntrain:, :]
    testy = label_all[ntrain:, :]
    return trainX, trainy, testX, testy


def generate_data(data, labels, batch_size):
    samples_per_epoch = data.shape[0]
    number_of_batches = samples_per_epoch // batch_size
    counter = 0
    while True:
        X_batch = data[batch_size * counter:batch_size * (counter + 1), :, :]
        y_batch = np.zeros(batch_size, 1+nHIwav)
        # Roll randomly
        rroll = np.random.random_integers(0, 15, batch_size)
        for ii in range(X_batch.shape[0]):
            if rroll[ii] == 0:
                y_batch[ii, :2] = labels[batch_size * ii:batch_size * (ii + 1), :]
                continue
            X_batch[ii, :, :] = np.roll(X_batch[ii, :, :], axis=0)
            y_batch[ii, -1] = 1
        counter += 1
        yield X_batch, y_batch

        # restart counter to yeild data in the next epoch as well
        if counter >= number_of_batches:
            counter = 0


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 10, 32
    n_timesteps, n_features = trainX.shape[1], trainX.shape[2]
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=16, kernel_size=5, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit_generator(
        generate_data(trainX, trainy, batch_size),
        steps_per_epoch=trainX.shape[0] // batch_size,
        validation_data=generate_data(testX, testy, batch_size*2),
        validation_steps=testX.shape[0] // batch_size * 2)

    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    #predict(model, scalewidths=10)
    return accuracy

# Gold standard in cross-validation
# from sklearn.model_selection import StratifiedKFold
# seed = 7
# np.random.seed(seed)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# for train, test in kfold.split(X, Y):
#     model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
#     # evaluate the model
#     scores = model.evaluate(X[test], Y[test], verbose=0)


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# Detect features in a dataset
def localise_features(repeats=3):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


# run the experiment
localise_features()

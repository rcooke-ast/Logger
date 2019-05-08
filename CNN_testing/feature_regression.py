import pdb
import numpy as np
from numpy import mean
from numpy import std
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from matplotlib import pyplot as plt


def predict(model):
    pdb.set_trace()
    predX, predy = make_gaussians(1000)
    pwidths = model.predict(predX, batch_size=None, verbose=0)
    plt.subplot(211)
    _, _, _ = plt.hist(predy[:, 0] - pwidths[0].flatten(), bins=30)
    plt.subplot(212)
    _, _, _ = plt.hist(predy[:, 1] - pwidths[1].flatten(), bins=30)
    plt.show()
    plt.subplot(211)
    plt.plot(predy[:, 0], pwidths[0].flatten(), 'bx')
    plt.subplot(212)
    plt.plot(predy[:, 1], pwidths[1].flatten(), 'bx')
    plt.show()


def make_gaussians(nmbr, wmin=0.5, wmax=2.5, amin=1.0, amax=2.0):
    widths = np.random.uniform(wmin, wmax, nmbr)
    ampls = np.random.uniform(amin, amax, nmbr)
    xval = np.linspace(-10.0, 10.0, 32)
    models = np.exp(-np.outer(0.5/widths**2, xval**2))
    targets = np.vstack((widths, ampls)).T
    #return models[:, :, np.newaxis], targets[:, :]
    return models, targets


# load the dataset, returns train and test X and y elements
def generate_dataset():
    # Generate some random Gaussians with different widths
    trainX, trainy = make_gaussians(10000)
    # load all test
    testX, testy = make_gaussians(1000)
    return trainX, trainy, testX, testy


# fit and evaluate a model
def evaluate_model_alt(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 10, 32
    model = Sequential()
    inshape = trainX.shape[1]
    #model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(inshape, 1)))
    # model.add(Conv1D(filters=16, kernel_size=5, activation='relu'))
    # model.add(Dropout(rate=0.5))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    model.add(Dense(64, input_dim=2, kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    predict(model)
    #pdb.set_trace()
    return accuracy


def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 10, 32
    inshape = trainX.shape[1]

    # This returns a tensor
    inputs = Input(shape=(inshape,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(16, activation='relu')(inputs)
    x = Dense(16, activation='relu')(x)
    pred_A = Dense(1, activation='linear', name='width')(x)
    pred_B = Dense(1, activation='linear', name='amplitude')(x)

    # Creates a model that includes the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=[pred_A, pred_B])
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(trainX, [trainy[:, 0], trainy[:, 1]], epochs=epochs, verbose=verbose)

    accuracy = model.evaluate(testX, [testy[:, 0], testy[:, 1]], batch_size=batch_size, verbose=0)
    predict(model)
    return accuracy


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# Detect features in a dataset
def detect_features(repeats=3):
    # load data
    trainX, trainy, testX, testy = generate_dataset()
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
detect_features()

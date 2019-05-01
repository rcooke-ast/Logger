import pdb
import numpy as np
from numpy import mean
from numpy import std
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from matplotlib import pyplot as plt


def predict(model):
    #pdb.set_trace()
    predX, predy = make_gaussians(1000)
    pwidths = model.predict(predX, batch_size=None, verbose=0)
    _, _, _ = plt.hist(predy-pwidths.flatten(), bins=30)
    plt.show()


def make_gaussians(nmbr, wmin=0.5, wmax=2.5):
    widths = np.random.uniform(wmin, wmax, nmbr)
    xval = np.linspace(-10.0, 10.0, 200)
    models = np.exp(-np.outer(0.5/widths**2, xval**2))
    return models, widths


# load the dataset, returns train and test X and y elements
def generate_dataset():
    # Generate some random Gaussians with different widths
    trainX, trainy = make_gaussians(10000)
    # load all test
    testX, testy = make_gaussians(1000)
    return trainX, trainy, testX, testy


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 10, 32
    model = Sequential()
    # model.add(Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=(n_timesteps,)))
    # model.add(Conv1D(filters=16, kernel_size=5, activation='relu'))
    # model.add(Dropout(rate=0.5))
    # model.add(Flatten())
    model.add(Dense(16, input_shape=(200,)))
    model.add(Activation('relu'))
    model.add(Dense(16, input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    predict(model)
    #pdb.set_trace()
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

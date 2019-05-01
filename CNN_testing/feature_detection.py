import pdb
import numpy as np
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


def predict(model, scalewidths=-1, wmin=0.5, wmax=2.5):
    if scalewidths == -1:
        scalewidths = wmax
    pdb.set_trace()
    predX, predy, widths = make_gaussians(1000, scalewidths=10, get_widths=True)
    #predy = np.round(predy)
    predy = predy
    preds = ((predy - wmin)/(scalewidths-wmin))* (wmax-wmin) + wmin
    vals = model.predict(predX, batch_size=None, verbose=0)
    predwidths = (preds*vals).sum(1)
    _, _, _ = plt.hist(widths-predwidths, bins=30)
    plt.show()


def make_gaussians(nmbr, scalewidths=-1, wmin=0.5, wmax=2.5, get_widths=False):
    if scalewidths == -1:
        scalewidths = wmax
    widths = np.random.uniform(wmin, wmax, nmbr)
    xval = np.linspace(-10.0, 10.0, 200)
    models = np.exp(-np.outer(0.5/widths**2, xval**2))
    labels = ((widths-wmin)/(wmax-wmin))*(scalewidths-wmin) + wmin
    if get_widths:
        return models[:, :, np.newaxis], labels[:, np.newaxis], widths
    else:
        return models[:, :, np.newaxis], labels[:, np.newaxis]


# load the dataset, returns train and test X and y elements
def generate_dataset(scalewidths=10):
    # Generate some random Gaussians with different widths
    trainX, trainy = make_gaussians(10000, scalewidths=scalewidths)
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = make_gaussians(1000, scalewidths=scalewidths)
    print(testX.shape, testy.shape)
#    trainy = np.round(trainy)
#    testy = np.round(testy)
    trainy = trainy
    testy = testy
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=16, kernel_size=5, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    predict(model, scalewidths=10)
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
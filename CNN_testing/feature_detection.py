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


def make_gaussians(nmbr, perturb=False):
    widths = np.random.randint(1, 6, nmbr)
    if perturb:
        widths = widths.astype(np.float) + np.random.uniform(-0.5, 0.5, nmbr)
    xval = np.linspace(-10.0, 10.0, 200)
    models = np.exp(-np.outer(0.5/widths**2, xval**2))
    return models[:, :, np.newaxis], widths[:, np.newaxis]


# load the dataset, returns train and test X and y elements
def generate_dataset():
    # Generate some random Gaussians with different widths
    trainX, trainy = make_gaussians(10000, perturb=True)
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = make_gaussians(1000, perturb=True)
    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = np.round(trainy) - 1
    testy = np.round(testy) - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
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
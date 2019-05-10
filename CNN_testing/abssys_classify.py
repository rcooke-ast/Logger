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


def make_gaussians(nmbr, wmin=0.5, wmax=2.5, amin=1.0, amax=2.0, addnoise=True):
    widths = np.random.uniform(wmin, wmax, nmbr)
    ampls = np.random.uniform(amin, amax, nmbr)
    xval = np.linspace(-10.0, 10.0, 16)
    models = np.exp(-np.outer(0.5/widths**2, xval**2))
    # Add noise
    noise = 0.0
    if addnoise:
        noise = np.random.normal(0.0, amax, models.shape)
    targets = np.vstack((widths, ampls)).T
    return models+noise, targets


def generate_dataset(nmbr=16384):
    ratio = 3.0  # This is the ratio of Lya to Lyb
    # Generate some random Gaussians with different widths
    trainXa, _ = make_gaussians(nmbr)
    trainXb, _ = make_gaussians(nmbr)
    label = np.random.random_integers(0, 1, nmbr)
    fact = ratio * label
    label = to_categorical(label)
    trainX = np.append((trainXa*fact[:, np.newaxis])[:, :, np.newaxis],
                       trainXb[:, :, np.newaxis], axis=2)
    return trainX, label


def predict(model, scalewidths=-1, wmin=0.5, wmax=2.5):
    if scalewidths == -1:
        scalewidths = wmax
    pdb.set_trace()
    predX, predy, widths = make_gaussians(1024)
    #predy = np.round(predy)
    predy = predy
    preds = ((predy - wmin)/(scalewidths-wmin)) * (wmax-wmin) + wmin
    vals = model.predict(predX, batch_size=None, verbose=0)
    predwidths = (preds*vals).sum(1)
    _, _, _ = plt.hist(widths-predwidths, bins=30)
    plt.show()


def generate_data(data, labels, batch_size):
    samples_per_epoch = data.shape[0]
    number_of_batches = samples_per_epoch // batch_size
    counter = 0
    while True:
        X_batch = data[batch_size * counter:batch_size * (counter + 1), :, :]
        y_batch = labels[batch_size * counter:batch_size * (counter + 1)]
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
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit_generator(
        generate_data(trainX, trainy, batch_size),
        steps_per_epoch=trainX.shape[0] // batch_size)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    #predict(model, scalewidths=10)
    return accuracy


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# Detect features in a dataset
def classify_features(repeats=3):
    # load data
    trainX, trainy = generate_dataset()
    testX, testy = generate_dataset()
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
classify_features()

import pdb
import numpy as np
from utilities import load_atomic
from numpy import mean
from numpy import std
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.layers import Dropout#, BatchNormalization

vpix = 2.5   # Size of each pixel in km/s
scalefact = np.log(1.0 + vpix/299792.458)
spec_len = 64  # Number of pixels to use in each segment
nHIwav = 15    # Number of lyman series lines to consider
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
    cntr_spec = 0
    while True:
        #[samples, speclen, lylinesspectra]
        #[batch_size, spec_len, nHIwav]
        #X_batch = data[batch_size * cntr_batch:batch_size * (cntr_batch + 1), :, :]
        indict = ({})
        for ll in range(nHIwav):
            X_batch = -1 * np.ones(data.shape[0], spec_len, nHIwav)
            indict['Ly{0:d}'.format(ll+1)] = X_batch.copy()
            for nn in range(nHIwav):
                nshft = int(np.round(np.log(HIwav[nn]/HIwav[ll])/scalefact))
                X_batch

        y_batch = np.zeros(batch_size, 1+nHIwav)
        cntr_batch += 1
        yield (indict, {'main_output': y_batch})

        # restart counter to yield data in the next epoch as well
        if cntr_batch >= number_of_batches:
            cntr_batch = 0


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 10, 32
    inputs = []
    concat_arr = []
    for ll in range(nHIwav):
        inputs.append(Input(shape=(spec_len, nHIwav), name='Ly{0:d}'.format(ll+1)))
        conv11 = Conv1D(filters=16, kernel_size=3, activation='relu')(inputs[-1])
        pool11 = MaxPooling1D(pool_size=2)(conv11)
        conv12 = Conv1D(filters=16, kernel_size=5, activation='relu')(pool11)
        drop11 = Dropout(rate=0.5)(conv12)
        pool12 = MaxPooling1D(pool_size=2)(drop11)
        concat_arr.append(Flatten()(pool12))
    # merge input models
    merge = concatenate(concat_arr)
    # interpretation model
    hidden1 = Dense(100, activation='relu')(merge)
    #hidden2 = Dense(100, activation='relu')(hidden1)
    output = Dense(1+nHIwav, activation='softmax', name='main_output')(hidden1)
    model = Model(inputs=inputs, outputs=output)
    # Summarize layers
    print(model.summary())
    # Plot graph
    plot_model(model, to_file='cnn_find_model.png')
    # Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit network
    model.fit_generator(
        generate_data(trainX, trainy, batch_size),
        steps_per_epoch=trainX.shape[0] // batch_size,
        validation_data=generate_data(testX, testy, batch_size*2),
        validation_steps=testX.shape[0] // batch_size * 2)

    # Eevaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
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
    pdb.set_trace()
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

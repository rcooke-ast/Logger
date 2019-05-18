
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
from keras.layers import Dropout, BatchNormalization

vpix = 2.5   # Size of each pixel in km/s
scalefact = np.log(1.0 + vpix/299792.458)
spec_len = 65  # Number of pixels to use in each segment (must be odd)
nHIwav = 15    # Number of lyman series lines to consider
atmdata = load_atomic(return_HIwav=False)
ww = np.where(atmdata["Ion"] == "1H_I")
HIwav = atmdata["Wavelength"][ww][3:]
HIfvl = atmdata["fvalue"][ww][3:]


def load_dataset(zem=3.0, snr=0, ftrain=0.75, numspec=20, epochs=10):
    zstr = "zem{0:.2f}".format(zem)
    sstr = "snr{0:d}".format(int(snr))
    extstr = "{0:s}_{1:s}_nspec{2:d}".format(zstr, sstr, numspec)
    fdata_all = np.load("train_data/cnn_qsospec_fluxspec_{0:s}_fluxonly.npy".format(extstr))
    IDlabel_all = np.load("train_data/cnn_qsospec_fluxspec_{0:s}_IDlabelonly.npy".format(extstr)).astype(np.int)
    Nlabel_all = np.load("train_data/cnn_qsospec_fluxspec_{0:s}_Nlabelonly.npy".format(extstr))
    blabel_all = np.load("train_data/cnn_qsospec_fluxspec_{0:s}_blabelonly.npy".format(extstr))
    ntrain = int(ftrain*fdata_all.shape[0])
    speccut = epochs*((fdata_all.shape[1]-spec_len)//epochs)
    trainX = fdata_all[:ntrain, -speccut:]
    trainy = IDlabel_all[:ntrain, -speccut:]
    trainN = Nlabel_all[:ntrain, -speccut:]
    trainb = blabel_all[:ntrain, -speccut:]
    testX = fdata_all[ntrain:, -speccut:]
    testy = IDlabel_all[ntrain:, -speccut:]
    testN = Nlabel_all[ntrain:, -speccut:]
    testb = blabel_all[ntrain:, -speccut:]
    print(trainX.shape[1], trainX.shape[1]//epochs, trainX.shape[1]%epochs)
    return trainX, trainy, trainN, trainb, testX, testy, testN, testb


def generate_data(data, IDlabels, Nlabels, blabels):
    cntr_spec = data.shape[1]-spec_len
    IDarr = np.arange(IDlabels.shape[0], dtype=np.int)
    while True:
        indict = ({})
        for ll in range(nHIwav):
            X_batch = np.ones((data.shape[0], spec_len, nHIwav))
            for nn in range(nHIwav):
                nshft = int(np.round(np.log(HIwav[nn]/HIwav[ll])/scalefact))
                lmin = cntr_spec+nshft
                lmax = lmin + spec_len
                if lmin < 0 or lmax >= data.shape[1]:
                    # Gone off the edge of the spectrum
                    continue
                X_batch[:, :, nn] = data[:, lmin:lmax]
            indict['Ly{0:d}'.format(ll + 1)] = X_batch.copy()
        ID_batch = np.zeros((IDlabels.shape[0], 1 + nHIwav))
        ID_batch[(IDarr, IDlabels[:, cntr_spec+(spec_len-1)//2],)] = 1
        N_batch = Nlabels[:, cntr_spec+(spec_len-1)//2]
        b_batch = blabels[:, cntr_spec+(spec_len-1)//2]
        outdict = {'ID_output': ID_batch, 'N_output': N_batch, 'b_output': b_batch}
        cntr_spec -= 1
        yield (indict, outdict)

        # restart counter to yield data in the next epoch as well
        if cntr_spec <= 0:
            cntr_spec = data.shape[1]-spec_len


# fit and evaluate a model
def evaluate_model(trainX, trainy, trainN, trainb,
                   testX, testy, testN, testb,
                   epochs=10, verbose=1):
    inputs = []
    concat_arr = []
    for ll in range(nHIwav):
        inputs.append(Input(shape=(spec_len, nHIwav), name='Ly{0:d}'.format(ll+1)))
        conv11 = Conv1D(filters=16, kernel_size=3, activation='relu')(inputs[-1])
        pool11 = MaxPooling1D(pool_size=2)(conv11)
        norm11 = BatchNormalization()(pool11)
        conv12 = Conv1D(filters=16, kernel_size=5, activation='relu')(norm11)
#        drop11 = Dropout(rate=0.5)(conv12)
        pool12 = MaxPooling1D(pool_size=2)(conv12)
        norm12 = BatchNormalization()(pool12)
        concat_arr.append(Flatten()(norm12))
    # merge input models
    merge = concatenate(concat_arr)
    # interpretation model
    #hidden2 = Dense(100, activation='relu')(hidden1)
    fullcon = Dense(100, activation='relu')(merge)
    ID_output = Dense(1+nHIwav, activation='softmax', name='ID_output')(fullcon)
    N_output = Dense(1, activation='linear', name='N_output')(fullcon)
    b_output = Dense(1, activation='linear', name='b_output')(fullcon)
    model = Model(inputs=inputs, outputs=[ID_output, N_output, b_output])
    # Summarize layers
    print(model.summary())
    # Plot graph
    plot_model(model, to_file='cnn_find_model.png')
    # Compile
    loss = {'ID_output': 'categorical_crossentropy',
            'N_output': 'mse',
            'b_output': 'mse'}
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    # Fit network
    model.fit_generator(
        generate_data(trainX, trainy, trainN, trainb),
        steps_per_epoch=(trainX.shape[1] - spec_len)//epochs,
        epochs=epochs, verbose=verbose,
        validation_data=generate_data(testX, testy, testN, testb),
        validation_steps=(testX.shape[1] - spec_len)//epochs)

    # Eevaluate model
    _, accuracy = model.evaluate_generator(generate_data(testX, testy, testN, testb),
                                           steps=(testX.shape[1] - spec_len)//epochs,
                                           verbose=0)
    print(accuracy)
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
def localise_features(repeats=3, epochs=10):
    # load data
    trainX, trainy, trainN, trainb, testX, testy, testN, testb = load_dataset(epochs=epochs)
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, trainN, trainb, testX, testy, testN, testb, epochs=epochs)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


# run the experiment
localise_features(epochs=10)

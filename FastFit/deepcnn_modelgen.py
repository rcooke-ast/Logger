
import os
import pdb
import numpy as np
from utilities import load_atomic
from contextlib import redirect_stdout
from numpy import mean, std
import tensorflow as tf
import keras.backend as K
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.layers import Dropout, BatchNormalization

# Limit the number of CPUs to use for training
#ncpus = 120
#K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=ncpus, inter_op_parallelism_threads=ncpus)))


vpix = 2.5   # Size of each pixel in km/s
scalefact = np.log(1.0 + vpix/299792.458)
spec_len = 129  # Number of pixels to use in each segment (must be odd)
nHIwav = 4    # Number of lyman series lines to consider
atmdata = load_atomic(return_HIwav=False)
ww = np.where(atmdata["Ion"] == "1H_I")
HIwav = atmdata["Wavelength"][ww][3:]
HIfvl = atmdata["fvalue"][ww][3:]


# Define custom loss
def mse_mask():
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        return K.mean(mask * K.square(y_pred - y_true), axis=-1)
    # Return a function
    return loss


def load_dataset(zem=3.0, snr=0, ftrain=2.0/2.25, numspec=25000, ispec=0, epochs=10):
    zstr = "zem{0:.2f}".format(zem)
    sstr = "snr{0:d}".format(int(snr))
    extstr = "{0:s}_{1:s}_nspec{2:d}_i{3:d}".format(zstr, sstr, numspec, ispec)
    # fdata_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_normalised_fluxonly.npy".format(extstr, nHIwav))
    # IDlabel_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_IDlabelonly.npy".format(extstr, nHIwav)).astype(np.int)
    # Nlabel_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_Nlabelonly.npy".format(extstr, nHIwav))
    # blabel_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_blabelonly.npy".format(extstr, nHIwav))
    # zlabel_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_zlabelonly.npy".format(extstr, nHIwav))
    fdata_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_normalised_fluxonly.npy".format(extstr, nHIwav))[:5000, :]
    IDlabel_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_IDlabelonly_vs0-ve5000.npy".format(extstr, nHIwav)).astype(np.int)
    Nlabel_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_Nlabelonly_vs0-ve5000.npy".format(extstr, nHIwav))
    blabel_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_blabelonly_vs0-ve5000.npy".format(extstr, nHIwav))
    zlabel_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_zlabelonly_vs0-ve5000.npy".format(extstr, nHIwav))
    ntrain = int(ftrain*fdata_all.shape[0])
    speccut = epochs*((fdata_all.shape[1]-spec_len)//epochs)
    trainX = fdata_all[:ntrain, -speccut:]
    trainy = IDlabel_all[:ntrain, -speccut:, 0]
    trainN = Nlabel_all[:ntrain, -speccut:, 0]
    trainz = zlabel_all[:ntrain, -speccut:, 0]
    trainb = blabel_all[:ntrain, -speccut:, 0]
    if False:
        pdb.set_trace()
        from matplotlib import pyplot as plt
        plt.plot(trainX[0, :], 'k-', drawstyle='steps')
        tlocs = np.where(trainz[0, :] == 1)[0]-1
        plt.vlines(tlocs, 0, 1, 'r', '-')
        plt.show()
    testX = fdata_all[ntrain:, -speccut:]
    testy = IDlabel_all[ntrain:, -speccut:, 0]
    testN = Nlabel_all[ntrain:, -speccut:, 0]
    testz = zlabel_all[ntrain:, -speccut:, 0]
    testb = blabel_all[ntrain:, -speccut:, 0]
    print(trainX.shape[1], trainX.shape[1]//epochs, trainX.shape[1]%epochs)
    return trainX, trainy, trainN, trainz, trainb, testX, testy, testN, testz, testb

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras see example from:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def generate_data(data, IDlabels, Nlabels, zlabels, blabels):
    shuff = np.arange(data.shape[1]-spec_len, dtype=np.int)
    np.random.shuffle(shuff)
    cntr_spec = 0
    IDarr = np.arange(IDlabels.shape[0], dtype=np.int)
    while True:
        indict = ({})
        for ll in range(nHIwav):
            X_batch = np.ones((data.shape[0], spec_len, nHIwav))
            for nn in range(nHIwav):
                nshft = int(np.round(np.log(HIwav[nn]/HIwav[ll])/scalefact))
                lmin = shuff[cntr_spec]+nshft
                lmax = lmin + spec_len
                if lmin < 0 or lmax >= data.shape[1]:
                    # Gone off the edge of the spectrum
                    continue
                X_batch[:, :, nn] = data[:, lmin:lmax]
            indict['Ly{0:d}'.format(ll + 1)] = X_batch.copy()
        midid = shuff[cntr_spec]+(spec_len-1)//2
        ID_batch = np.zeros((IDlabels.shape[0], 1 + nHIwav))
        ID_batch[(IDarr, IDlabels[:, midid],)] = 1
        N_batch = Nlabels[:, midid]
        z_batch = zlabels[:, midid]
        b_batch = blabels[:, midid]
        outdict = {'ID_output': ID_batch, 'N_output': N_batch, 'z_output': z_batch, 'b_output': b_batch}
        cntr_spec += 1
        yield (indict, outdict)

        # restart counter to yield data in the next epoch as well
        if cntr_spec >= data.shape[1]-spec_len:
            cntr_spec = 0


# fit and evaluate a model
def evaluate_model(trainX, trainy, trainN, trainz, trainb,
                   testX, testy, testN, testz, testb,
                   epochs=10, verbose=1):
    filepath = os.path.dirname(os.path.abspath(__file__))
    model_name = "/fit_data/model_nLy{0:d}_speclen{1:d}".format(nHIwav, spec_len)
    inputs = []
    concat_arr = []
    for ll in range(nHIwav):
        inputs.append(Input(shape=(spec_len, nHIwav), name='Ly{0:d}'.format(ll+1)))
        conv11 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs[-1])
        conv12 = Conv1D(filters=64, kernel_size=3, activation='relu')(conv11)
        pool1  = MaxPooling1D(pool_size=2)(conv12)
        conv21 = Conv1D(filters=128, kernel_size=3, activation='relu')(pool1)
        conv22 = Conv1D(filters=128, kernel_size=3, activation='relu')(conv21)
        pool2  = MaxPooling1D(pool_size=2)(conv22)
        conv31 = Conv1D(filters=256, kernel_size=3, activation='relu')(pool2)
        conv32 = Conv1D(filters=256, kernel_size=3, activation='relu')(conv31)
        pool3  = MaxPooling1D(pool_size=2)(conv32)
        conv41 = Conv1D(filters=512, kernel_size=3, activation='relu')(pool3)
        conv42 = Conv1D(filters=512, kernel_size=3, activation='relu')(conv41)
        pool4  = MaxPooling1D(pool_size=2)(conv42)
        concat_arr.append(Flatten()(pool4))
    # merge input models
    merge = concatenate(concat_arr)
    # interpretation model
    #hidden2 = Dense(100, activation='relu')(hidden1)
    fullcon1 = Dense(4096, activation='relu')(merge)
    fullcon2 = Dense(4096, activation='relu')(fullcon1)
    ID_output = Dense(1+nHIwav, activation='softmax', name='ID_output')(fullcon2)
    N_output = Dense(1, activation='linear', name='N_output')(fullcon2)
    z_output = Dense(1, activation='linear', name='z_output')(fullcon2)
    b_output = Dense(1, activation='linear', name='b_output')(fullcon2)
    model = Model(inputs=inputs, outputs=[ID_output, N_output, z_output, b_output])
    # Summarize layers
    with open(filepath + model_name + '.summary', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    # Plot graph
    pngname = filepath + model_name + '.png'
    plot_model(model, to_file=pngname)
    # Compile
    loss = {'ID_output': 'categorical_crossentropy',
            'N_output': mse_mask(),
            'z_output': mse_mask(),
            'b_output': mse_mask()}
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    # Initialise callbacks
    ckp_name = filepath + model_name + '.hdf5'
    csv_name = filepath + model_name + '.log'
    checkpointer = ModelCheckpoint(filepath=ckp_name, verbose=1, save_best_only=True)
    csv_logger = CSVLogger(csv_name, append=True)
    # Fit network
    model.fit_generator(
        generate_data(trainX, trainy, trainN, trainz, trainb),
        steps_per_epoch=(trainX.shape[1] - spec_len)//epochs,
        epochs=epochs, verbose=verbose,
        callbacks=[checkpointer, csv_logger],
        validation_data=generate_data(testX, testy, testN, testz, testb),
        validation_steps=(testX.shape[1] - spec_len)//epochs)

    # Evaluate model
#    _, accuracy
    accuracy = model.evaluate_generator(generate_data(testX, testy, testN, testz, testb),
                                        steps=(testX.shape[1] - spec_len)//epochs,
                                        verbose=0)
    #pdb.set_trace()
    return accuracy, model.metrics_names

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
    keys = scores.keys()
    for ii in keys:
        m, s = mean(scores[ii]), std(scores[ii])
        print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# Detect features in a dataset
def localise_features(repeats=3, epochs=10):
    # load data
    trainX, trainy, trainN, trainz, trainb,\
    testX, testy, testN, testz, testb = load_dataset(epochs=epochs)
    # repeat experiment
    allscores = dict({})
    for r in range(repeats):
        scores, names = evaluate_model(trainX, trainy, trainN, trainz, trainb,
                                       testX, testy, testN, testz, testb, epochs=epochs)
        if r == 0:
            for name in names:
                allscores[name] = []
        for ii, name in enumerate(names):
            if '_acc' in name:
                allscores[name].append(scores[ii] * 100.0)
                print('%s >#%d: %.3f' % (name, r + 1, allscores[name][-1]))
            else:
                print('%s >#%d: %.3f' % (name, r + 1, scores[ii]))
    # summarize results
    summarize_results(allscores)


# run the experiment
localise_features(epochs=10)

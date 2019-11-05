import os
import pdb
import time
import pickle
import numpy as np
from pyigm.fN.fnmodel import FNModel
from pyigm.fN.mockforest import monte_HIcomp
from scipy.special import wofz
from utilities import generate_wave, rebin_subpix, load_atomic
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.python.client import device_lib
import keras.backend as K
from keras.utils import plot_model, multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout, Flatten
from keras import regularizers
from contextlib import redirect_stdout

velstep = 2.5    # Pixel size in km/s
spec_len = 256
spec_ext = 64
nHIwav = 1    # Number of lyman series lines to consider
atmdata = load_atomic(return_HIwav=False)
ww = np.where(atmdata["Ion"] == "1H_I")
HIwav = atmdata["Wavelength"][ww][3:]
HIfvl = atmdata["fvalue"][ww][3:]


# Define custom loss
def mse_mask():
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        epsilon = K.ones_like(y_true[0,:])*0.00001
        return K.mean( (y_true/(y_true+epsilon)) * K.square(y_pred - y_true), axis=-1)
        #return K.mean(K.square(y_pred - y_true), axis=-1)
    # Return a function
    return loss


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def save_obj(obj, dirname):
    with open(dirname + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(dirname):
    with open(dirname + '.pkl', 'rb') as f:
        return pickle.load(f)


def hyperparam_orig(mnum):
    """Generate a random set of hyper parameters

    mnum (int): Model index number
    """
    # Define all of the allowed parameter space
    allowed_hpars = dict(learning_rate      = [0.001],
                         lr_decay           = [0.0],
                         l2_regpen          = [0.0],
                         dropout_prob       = [0.0],
                         num_epochs         = [10],
                         batch_size         = [1000],
                         num_batch_train    = [200],
                         num_batch_validate = [20],
                         # Number of filters in each convolutional layer
                         conv_filter_1 = [128],
                         conv_filter_2 = [128],
                         conv_filter_3 = [128],
                         # Kernel size
                         conv_kernel_1 = [32],
                         conv_kernel_2 = [32],
                         conv_kernel_3 = [32],
                         # Stride of each kernal
                         conv_stride_1 = [1],
                         conv_stride_2 = [1],
                         conv_stride_3 = [1],
                         # Pooling kernel size
                         pool_kernel_1 = [2],
                         pool_kernel_2 = [2],
                         pool_kernel_3 = [2],
                         # Pooling stride
                         pool_stride_1 = [1, 2, 3],
                         pool_stride_2 = [1, 2, 3],
                         pool_stride_3 = [1, 2, 3],
                         # Fully connected layers
                         fc1_neurons   = [4096],
                         fc2_N_neurons = [32, 64, 128, 256],
                         fc2_z_neurons = [32, 64, 128, 256],
                         fc2_b_neurons = [32, 64, 128, 256],
                         )
    # Generate dictionary of values
    hyperpar = dict({})
    for key in allowed_hpars.keys():
        hyperpar[key] = np.random.choice(allowed_hpars[key])
    # Save these parameters and return the hyperpar
    save_obj(hyperpar, 'fit_data/multi/model_{0:03d}'.format(mnum))
    return hyperpar


def hyperparam(mnum):
    """Generate a random set of hyper parameters

    mnum (int): Model index number
    """
    # Define all of the allowed parameter space
    allowed_hpars = dict(learning_rate      = [0.0005, 0.0007, 0.0010, 0.0030, 0.0050, 0.0070, 0.0100],
                         lr_decay           = [0.0, 1.0],
                         l2_regpen          = [0.0, 0.00001, 0.00010, 0.00100, 0.00500, 0.01000],
                         dropout_prob       = [0.0, 0.01, 0.02, 0.05],
                         num_epochs         = [30, 50, 100],
                         batch_size         = [100, 500, 1000, 2000, 5000],
                         num_batch_train    = [128, 256, 512, 1024],
                         num_batch_validate = [32, 64, 128],
                         # Number of filters in each convolutional layer
                         conv_filter_1 = [80, 96, 128, 192, 256],
                         conv_filter_2 = [80, 96, 128, 192, 256],
                         conv_filter_3 = [80, 96, 128, 192, 256],
                         # Kernel size
                         conv_kernel_1 = [20, 22, 24, 26, 28, 32, 40, 48, 54],
                         conv_kernel_2 = [10, 14, 16, 20, 24, 28, 32, 34],
                         conv_kernel_3 = [10, 14, 16, 20, 24, 28, 32, 34],
                         # Stride of each kernal
                         conv_stride_1 = [1, 2, 4, 6],
                         conv_stride_2 = [1, 2, 4, 6],
                         conv_stride_3 = [1, 2, 4, 6],
                         # Pooling kernel size
                         pool_kernel_1 = [2, 3, 4, 6],
                         pool_kernel_2 = [2, 3, 4, 6],
                         pool_kernel_3 = [2, 3, 4, 6],
                         # Pooling stride
                         pool_stride_1 = [1, 2, 3],
                         pool_stride_2 = [1, 2, 3],
                         pool_stride_3 = [1, 2, 3],
                         # Fully connected layers
                         fc1_neurons   = [256, 512, 1024, 2048],
                         fc2_N_neurons = [32, 64, 128, 256],
                         fc2_z_neurons = [32, 64, 128, 256],
                         fc2_b_neurons = [32, 64, 128, 256],
                         )
    # Generate dictionary of values
    hyperpar = dict({})
    for key in allowed_hpars.keys():
        hyperpar[key] = np.random.choice(allowed_hpars[key])
    # Save these parameters and return the hyperpar
    save_obj(hyperpar, 'fit_data/multi/model_{0:03d}'.format(mnum))
    return hyperpar


def load_dataset(zem=3.0, snr=0, ftrain=2.0/2.25, numspec=25000, ispec=0):
    zstr = "zem{0:.2f}".format(zem)
    sstr = "snr{0:d}".format(int(snr))
    extstr = "{0:s}_{1:s}_nspec{2:d}_i{3:d}".format(zstr, sstr, numspec, ispec)
    wmin = HIwav[nHIwav]*(1.0+zem)
    wdata_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_wave.npy".format(extstr))
    wuse = np.where(wdata_all > wmin)[0]
    fdata_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_normalised_fluxonly.npy".format(extstr))[:5000, wuse]
    Nlabel_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_Nlabelonly_vs0-ve5000.npy".format(extstr, nHIwav))[:, wuse, :]
    blabel_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_blabelonly_vs0-ve5000.npy".format(extstr, nHIwav))[:, wuse, :]
    zlabel_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_zlabelonly_vs0-ve5000.npy".format(extstr, nHIwav))[:, wuse, :]
    ntrain = int(ftrain*fdata_all.shape[0])
    trainX = fdata_all[:ntrain, :]
    trainN = Nlabel_all[:ntrain, :, 0]
    trainz = zlabel_all[:ntrain, :, 0]
    trainb = blabel_all[:ntrain, :, 0]
    if False:
        pdb.set_trace()
        from matplotlib import pyplot as plt
        plt.plot(trainX[0, :], 'k-', drawstyle='steps')
        tlocs = np.where(trainz[0, :] == 1)[0]-1
        plt.vlines(tlocs, 0, 1, 'r', '-')
        plt.show()
    testX = fdata_all[ntrain:, -speccut:]
    testN = Nlabel_all[ntrain:, -speccut:, 0]
    testz = zlabel_all[ntrain:, -speccut:, 0]
    testb = blabel_all[ntrain:, -speccut:, 0]
    print(trainX.shape[1], trainX.shape[1]//epochs, trainX.shape[1]%epochs)
    return trainX, trainN, trainz, trainb, testX, testN, testz, testb


def yield_data(data, Nlabels, zlabels, blabels, batch_sz, maskval=0.0):
    cntr_batch = 0
    cenpix = (spec_len+spec_ext)//2
    ll = np.arange(batch_sz).repeat(spec_len)
    while True:
        indict = ({})
        pertrb_s = np.random.randint(0, data.shape[0], batch_sz)
        pertrb_w = np.random.randint(cenpix, data.shape[0]-cenpix, batch_sz)
        pp = pertrb.reshape((batch_sz, 1)).repeat(spec_len, axis=1) + np.arange(spec_len)
        X_batch = data[ll+cntr_batch, pp.flatten()].reshape((batch_sz, spec_len, 1))
        indict['input_1'] = X_batch.copy()
        z_batch = spec_len//2 - cenpix + pertrb.copy()
        # Extract the relevant bits of information
        yld_N = Nlabels[cntr_batch:cntr_batch+batch_sz]
        yld_z = z_batch
        yld_b = blabels[cntr_batch:cntr_batch+batch_sz]
        # Mask
        if True:
            wmsk = np.where(X_batch[:, spec_len//2, 0] > 0.95)
            yld_N[wmsk] = maskval
            yld_z[wmsk] = maskval  # Note, this will mask true zeros in the yld_z array
            yld_b[wmsk] = maskval
        # Store output
        outdict = {'output_N': yld_N,
                   'output_z': yld_z,
                   'output_b': yld_b}

#        pdb.set_trace()
        yield (indict, outdict)

        cntr_batch += batch_sz
        if cntr_batch >= data.shape[0]-batch_sz:
            cntr_batch = 0


def build_model_simple(hyperpar):
    # Extract parameters
    fc1_neurons = hyperpar['fc1_neurons']
    fc2_N_neurons = hyperpar['fc2_N_neurons']
    fc2_b_neurons = hyperpar['fc2_b_neurons']
    fc2_z_neurons = hyperpar['fc2_z_neurons']
    conv1_kernel = hyperpar['conv_kernel_1']
    conv2_kernel = hyperpar['conv_kernel_2']
    conv3_kernel = hyperpar['conv_kernel_3']
    conv1_filter = hyperpar['conv_filter_1']
    conv2_filter = hyperpar['conv_filter_2']
    conv3_filter = hyperpar['conv_filter_3']
    conv1_stride = hyperpar['conv_stride_1']
    conv2_stride = hyperpar['conv_stride_2']
    conv3_stride = hyperpar['conv_stride_3']
    pool1_kernel = hyperpar['pool_kernel_1']
    pool2_kernel = hyperpar['pool_kernel_2']
    pool3_kernel = hyperpar['pool_kernel_3']
    pool1_stride = hyperpar['pool_stride_1']
    pool2_stride = hyperpar['pool_stride_2']
    pool3_stride = hyperpar['pool_stride_3']

    # Build model
    # Shape is (batches, steps, channels)
    # For example, a 3-color 1D image of side 100 pixels, dealt in batches of 32 would have a shape=(32,100,3)
    input_1 = Input(shape=(spec_len, 1), name='input_1')
    conv1 = Conv1D(filters=conv1_filter, kernel_size=(conv1_kernel,), strides=(conv1_stride,), activation='relu')(input_1)
    pool1 = MaxPooling1D(pool_size=(pool1_kernel,), strides=(pool1_stride,))(conv1)
    conv2 = Conv1D(filters=conv2_filter, kernel_size=(conv2_kernel,), strides=(conv2_stride,), activation='relu')(pool1)
    pool2 = MaxPooling1D(pool_size=(pool2_kernel,), strides=(pool2_stride,))(conv2)
    conv3 = Conv1D(filters=conv3_filter, kernel_size=(conv3_kernel,), strides=(conv3_stride,), activation='relu')(pool2)
    pool3 = MaxPooling1D(pool_size=(pool3_kernel,), strides=(pool3_stride,))(conv3)
    flatlay = Flatten()(pool3)

    # Interpretation model
    regpen = hyperpar['l2_regpen']
    fullcon1 = Dense(fc1_neurons, activation='relu', kernel_regularizer=regularizers.l2(regpen))(flatlay)
    drop1 = Dropout(hyperpar['dropout_prob'])(fullcon1)
    # Second fully connected layer
    fullcon2_N = Dense(fc2_N_neurons, activation='relu', kernel_regularizer=regularizers.l2(regpen))(drop1)
    drop2_N = Dropout(hyperpar['dropout_prob'])(fullcon2_N)
    fullcon2_z = Dense(fc2_z_neurons, activation='relu', kernel_regularizer=regularizers.l2(regpen))(drop1)
    drop2_z = Dropout(hyperpar['dropout_prob'])(fullcon2_z)
    fullcon2_b = Dense(fc2_b_neurons, activation='relu', kernel_regularizer=regularizers.l2(regpen))(drop1)
    drop2_b = Dropout(hyperpar['dropout_prob'])(fullcon2_b)
    output_N = Dense(1, activation='linear', name='output_N')(drop2_N)
    output_z = Dense(1, activation='linear', name='output_z')(drop2_z)
    output_b = Dense(1, activation='linear', name='output_b')(drop2_b)
    model = Model(inputs=[input_1], outputs=[output_N, output_z, output_b])
    return model


# fit and evaluate a model
def evaluate_model(trainX, trainN, trainz,  trainb,
                   testX, testN, testz, testb, hyperpar,
                   mnum, epochs=10, verbose=1):
    #yield_data(trainX, trainN, trainb)
    #assert(False)
    filepath = os.path.dirname(os.path.abspath(__file__))
    model_name = '/fit_data/multi/model_{0:03d}'.format(mnum)
    ngpus = len(get_available_gpus())
    # Construct network
    if ngpus > 1:
        model = build_model_simple(hyperpar)
        # Make this work on multiple GPUs
        gpumodel = multi_gpu_model(model, gpus=ngpus)
    else:
        gpumodel = build_model_simple(hyperpar)

    # Summarize layers
    summary = False
    if summary:
        with open(filepath + model_name + '.summary', 'w') as f:
            with redirect_stdout(f):
                model.summary()
    # Plot graph
    plotit = False
    if plotit:
        pngname = filepath + model_name + '.png'
        plot_model(model, to_file=pngname)
    # Compile
    masking = True
    if masking:
        loss = {'output_N': mse_mask(),
                'output_z': mse_mask(),
                'output_b': mse_mask()}
    else:
        loss = {'output_N': 'mse',
                'output_z': 'mse',
                'output_b': 'mse'}
    decay = hyperpar['lr_decay']*hyperpar['learning_rate']/hyperpar['num_epochs']
    optadam = Adam(lr=hyperpar['learning_rate'], decay=decay)
    gpumodel.compile(loss=loss, optimizer=optadam, metrics=['mean_squared_error'])
    # Initialise callbacks
    ckp_name = filepath + model_name + '.hdf5'
    sav_name = filepath + model_name + '_save.hdf5'
    csv_name = filepath + model_name + '.log'
    checkpointer = ModelCheckpoint(filepath=ckp_name, verbose=1, save_best_only=True)
    csv_logger = CSVLogger(csv_name, append=True)
    # Fit network
    gpumodel.fit_generator(
        yield_data(trainX, trainN, trainz, trainb, hyperpar['batch_size']),
        steps_per_epoch=hyperpar['num_batch_train'],  # Total number of batches (i.e. num data/batch size)
        epochs=epochs, verbose=verbose,
        callbacks=[checkpointer, csv_logger],
        validation_data=yield_data(testX, testN, testz, testb, hyperpar['batch_size']),
        validation_steps=hyperpar['num_batch_validate'])

    gpumodel.save(sav_name)

    # Evaluate model
#    _, accuracy
    accuracy = gpumodel.evaluate_generator(yield_data(testX, testN, testz, testb, hyperpar['batch_size']),
                                           steps=testX.shape[0],
                                           verbose=0)
    return accuracy, gpumodel.metrics_names


def restart_model(model_name, trainX, trainN, trainb,
                   testX, testN, testb,
                   epochs=10, verbose=1, masking=True):
    filepath = os.path.dirname(os.path.abspath(__file__))+'/'
    # Load model
    loadname = filepath + 'fit_data/' + model_name + '.hdf5'
    model = load_model(loadname, compile=False)
    # Make this work on multiple GPUs
    gpumodel = multi_gpu_model(model, gpus=4)
    # Compile model
    if masking:
        loss = {'N_output': mse_mask(),
                'z_output': mse_mask(),
                'b_output': mse_mask()}
    else:
        loss = {'N_output': 'mse',
                'z_output': 'mse',
                'b_output': 'mse'}
    gpumodel.compile(loss=loss, optimizer='adam', metrics=['mean_squared_error'])
    # Initialise callbacks
    ckp_name = filepath + model_name + '_chkp_restart.hdf5'
    sav_name = filepath + model_name + '_save_restart.hdf5'
    csv_name = filepath + model_name + '_restart.log'
    checkpointer = ModelCheckpoint(filepath=ckp_name, verbose=1, save_best_only=True)
    csv_logger = CSVLogger(csv_name, append=True)
    # Fit network
    gpumodel.fit_generator(
        yield_data(trainX, trainN, trainb),
        steps_per_epoch=2000,  # Total number of batches (i.e. num data/batch size)
        epochs=epochs, verbose=verbose,
        callbacks=[checkpointer, csv_logger],
        validation_data=yield_data(testX, testN, testb),
        validation_steps=200)

    gpumodel.save(sav_name)

    # Evaluate model
    #    _, accuracy
    accuracy = gpumodel.evaluate_generator(yield_data(testX, testN, testb),
                                           steps=testX.shape[0],
                                           verbose=0)
    return accuracy, gpumodel.metrics_names


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
        m, s = np.mean(scores[ii]), np.std(scores[ii])
        print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# Detect features in a dataset
def localise_features(mnum, repeats=3, restart=False):
    # Generate hyperparameters
    hyperpar = hyperparam(mnum)
    # load data
    trainX, trainN, trainz, trainb,\
    testX, testN, testz, testb = load_dataset()
    # repeat experiment
    allscores = dict({})
    for r in range(repeats):
        if restart:
            model_name = 'svoigt_speclen256_masked_save'
            scores, names = restart_model(model_name, trainX, trainN, trainb,
                                          testX, testN, testb)
        else:
            scores, names = evaluate_model(trainX, trainN, trainz, trainb,
                                           testX, testN, testz, testb, hyperpar, mnum, epochs=hyperpar['num_epochs'])
        if r == 0:
            for name in names:
                allscores[name] = []
        for ii, name in enumerate(names):
            allscores[name].append(scores[ii] * 100.0)
            if '_acc' in name:
                print('%s >#%d: %.3f' % (name, r + 1, allscores[name][-1]))
            else:
                print('%s >#%d: %.3f' % (name, r + 1, scores[ii]))
    # Summarize results
    summarize_results(allscores)

# Set the number of epochs
if False:
    # Generate data
    generate_dataset()
else:
    # Once the data exist, run the experiment
    m_init = 0
    mnum = m_init
    while True:
        try:
            localise_features(mnum, repeats=1, restart=False)
        except ValueError:
            continue
        mnum += 1
        if mnum >= m_init+1000:
            break

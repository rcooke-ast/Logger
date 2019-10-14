import pdb
import numpy as np
from pyigm.fN.fnmodel import FNModel
from pyigm.fN.mockforest import monte_HIcomp
from scipy.special import wofz
from utilities import generate_wave, rebin_subpix
from matplotlib import pyplot as plt
velstep = 2.5    # Pixel size in km/s


def voigt(wave, params):
    p0, p1, p2 = params
    lam, fvl, gam = 1215.6701, 0.4164, 6.265E8
    cold = 10.0**p0
    zp1=p1+1.0
    wv=lam*1.0e-8
    bl=p2*wv/2.99792458E5
    a=gam*wv*wv/(3.76730313461770655E11*bl)
    cns=wv*wv*fvl/(bl*2.002134602291006E12)
    cne=cold*cns
    ww=(wave*1.0e-8)/zp1
    v=wv*ww*((1.0/ww)-(1.0/wv))/bl
    tau = cne*wofz(v + 1j * a).real
    return np.exp(-1.0*tau)


def make_spectra(NHI, zabs, bval, nsubpix=10, szstr=None, numspec=1):
    if szstr is None:
        szstr = 256+64
    nspec = NHI.size
    vprofs = np.zeros((szstr, NHI.size))
    for ss in range(nspec):
        if ss%10000 == 0:
            print(ss, nspec)
        lam = 1215.6701*(1+zabs[ss])
        wmin = lam * (1 - 200*velstep/299792.458)
        wmax = lam * (1 + 200*velstep/299792.458)
        wave, subwave = generate_wave(wavemin=wmin, wavemax=wmax, velstep=velstep, nsubpix=nsubpix)
        amn = np.argmin(np.abs(subwave-lam))
        flux = voigt(subwave[amn-nsubpix*szstr//2: amn+nsubpix*szstr//2], [NHI[ss], zabs[ss], bval[ss]])
        vprofs[:, ss] = rebin_subpix(flux, nsubpix=nsubpix)
    suffix = "zem3_nsubpix{0:d}_numspec{1:d}".format(nsubpix, numspec)
    np.save("train_data/svoigt_prof_{0:s}".format(suffix), vprofs)
    np.save("train_data/svoigt_Nval_{0:s}".format(suffix), NHI)
    np.save("train_data/svoigt_bval_{0:s}".format(suffix), bval/velstep)
    return


def generate_dataset(zem=3.0, snr=0, seed=1234, ftrain=0.9, nsubpix=10, numspec=1000, epochs=10):
    # Setup params
    rstate = np.random.RandomState(seed)
    zmin = 1026.0*(1+zem)/1215.6701 - 1

    # Get the CDDF
    NHIp = np.array([12.0, 15.0, 17.0, 18.0, 20.0, 21.0, 21.5, 22.0])
    sply = np.array([-9.72, -14.41, -17.94, -19.39, -21.28, -22.82, -23.95, -25.50])
    params = dict(sply=sply)
    NHI, zabs, bval = np.array([]), np.array([]), np.array([])
    for ss in range(numspec):
        fN_model = FNModel('Hspline', pivots=NHIp, param=params, zmnx=(2., 5.))
        HI_comps = monte_HIcomp((zmin, zem), fN_model, NHI_mnx=(12., 18.), rstate=rstate)

        # Extract/store information
        NHI = np.append(NHI, HI_comps['lgNHI'].data)
        zabs = np.append(zabs, HI_comps['z'].data)
        bval = np.append(bval, HI_comps['bval'].value)
    make_spectra(NHI, zabs, bval, nsubpix=nsubpix, numspec=numspec)


def load_dataset(zem=3.0, snr=0, ftrain=0.9, epochs=10):
    zstr = "zem{0:.2f}".format(zem)
    fdata = np.load("svoigt_prof_zem3_nsubpix10_numspec1000.npy")
    Nlabel = np.load("svoigt_Nval_zem3_nsubpix10_numspec1000.npy")
    blabel = np.load("svoigt_bval_zem3_nsubpix10_numspec1000.npy")
    zlabel = ((256+64)//2)*np.ones(Nlabel.size)
    ntrain = int(ftrain*fdata.shape[0])
    # Select the training data
    trainX = fdata[:ntrain, :]
    trainN = Nlabel[:ntrain, :]
    trainz = zlabel[:ntrain, :]
    trainb = blabel[:ntrain, :]
    # Select the test data
    testX = fdata[ntrain:, :]
    testN = Nlabel[ntrain:, :]
    testz = zlabel[ntrain:, :]
    testb = blabel[ntrain:, :]
    print(trainX.shape[1], trainX.shape[1]//epochs, trainX.shape[1]%epochs)
    return trainX, trainN, trainz, trainb, testX, testN, testz, testb


# fit and evaluate a model
def evaluate_model(trainX, trainy, trainN, trainz, trainb,
                   testX, testy, testN, testz, testb,
                   epochs=10, verbose=1):
#    generate_data_test(testX, testy, testN, testz, testb)
#    assert(False)
    filepath = os.path.dirname(os.path.abspath(__file__))
    model_name = "/fit_data/model_nLy{0:d}_speclen{1:d}multi4".format(nHIwav, spec_len[0])
    inputs = []
    concat_arr = []
    kernsz = [15, 9, 5, 3]
    for ll in range(len(spec_len)):
        inputs.append(Input(shape=(spec_len[ll],1), name='Lya_{0:d}'.format(ll+1)))
        conv11 = Conv1D(filters=64, kernel_size=kernsz[ll], activation='relu')(inputs[-1])
#        conv12 = Conv1D(filters=64, kernel_size=16, activation='relu')(conv11)
#        pool1  = MaxPooling1D(pool_size=2)(conv11)
        conv21 = Conv1D(filters=128, kernel_size=kernsz[ll], activation='relu')(conv11)
#        conv22 = Conv1D(filters=128, kernel_size=16, activation='relu')(conv21)
#        pool2  = MaxPooling1D(pool_size=2)(conv21)
        conv31 = Conv1D(filters=256, kernel_size=kernsz[ll], activation='relu')(conv21)
#        conv32 = Conv1D(filters=256, kernel_size=3, activation='relu')(conv31)
#        pool3  = MaxPooling1D(pool_size=2)(conv31)
#        conv41 = Conv1D(filters=512, kernel_size=3, activation='relu')(pool3)
#        conv42 = Conv1D(filters=512, kernel_size=3, activation='relu')(conv41)
#        pool4  = MaxPooling1D(pool_size=2)(conv42)
        concat_arr.append(Flatten()(conv31))
        #concat_arr.append(pool3)
    if nHIwav == 1 and len(spec_len)==1:
        merge = concat_arr[0]
    else:
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
        generate_data_multilen(trainX, trainy, trainN, trainz, trainb),
        steps_per_epoch=(trainX.shape[1] - spec_len[0])//epochs,
        epochs=epochs, verbose=verbose,
        callbacks=[checkpointer, csv_logger],
        validation_data=generate_data(testX, testy, testN, testz, testb),
        validation_steps=(testX.shape[1] - spec_len[0])//epochs)

    # Evaluate model
#    _, accuracy
    accuracy = model.evaluate_generator(generate_data_multilen(testX, testy, testN, testz, testb),
                                        steps=(testX.shape[1] - spec_len[0])//epochs,
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
    # Summarize results
    summarize_results(allscores)

# Set the number of epochs
epochs = 10

# Generate data
generate_dataset(epochs=epochs)

# Once the data exist, run the experiment
if False:
    localise_features(epochs=10)

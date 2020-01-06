import numpy as np
from matplotlib import pyplot as plt
from utilities import load_atomic

velstep = 2.5    # Pixel size in km/s
nHIwav = 1       # Number of lyman series lines to consider
zmskrng = 10     # Number of pixels away from a line centre that we wish to include in MSE
atmdata = load_atomic(return_HIwav=False)
ww = np.where(atmdata["Ion"] == "1H_I")
HIwav = atmdata["Wavelength"][ww][3:]
HIfvl = atmdata["fvalue"][ww][3:]


def plot_labels(zem=3.0, snr=0, ftrain=2.0/2.25, numspec=25000, ispec=0):
    zstr = "zem{0:.2f}".format(zem)
    sstr = "snr{0:d}".format(int(snr))
    extstr = "{0:s}_{1:s}_nspec{2:d}_i{3:d}".format(zstr, sstr, numspec, ispec)
    wmin = HIwav[nHIwav]*(1.0+zem)
    wdata_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_wave.npy".format(extstr))
    wuse = np.where(wdata_all > wmin)[0]
    fdata_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_normalised_fluxonly.npy".format(extstr))[:5000, wuse]
    Nlabel_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_Nlabelonly_vs0-ve5000_fixz.npy".format(extstr, nHIwav))[:, wuse, :]
    blabel_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_blabelonly_vs0-ve5000_fixz.npy".format(extstr, nHIwav))[:, wuse, :]
    zlabel_all = np.load("label_data/cnn_qsospec_fluxspec_{0:s}_nLy{1:d}_zlabelonly_vs0-ve5000_fixz.npy".format(extstr, nHIwav))[:, wuse, :]
    ntrain = int(ftrain*fdata_all.shape[0])
    trainX = fdata_all[:ntrain, :]
    trainN = Nlabel_all[:ntrain, :, :]
    trainz = zlabel_all[:ntrain, :, :]
    trainb = blabel_all[:ntrain, :, :]
    # Plot the data and labels
    fig, axs = plt.subplots(4, 1, sharex='all')
    axs[0].plot(trainX[0, :], 'k-', drawstyle='steps')
    axs[1].plot(trainN[0, :, 0], 'k-', drawstyle='steps')
    axs[1].plot(trainN[0, :, 1], 'r-', drawstyle='steps')
    axs[2].plot(trainz[0, :, 0], 'k-', drawstyle='steps')
    axs[2].plot(trainz[0, :, 1], 'r-', drawstyle='steps')
    axs[3].plot(trainb[0, :, 0], 'k-', drawstyle='steps')
    axs[3].plot(trainb[0, :, 1], 'r-', drawstyle='steps')
    plt.show()
    return

plot_labels()


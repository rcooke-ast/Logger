# This script reads the spectra generated with
# cnn_create_training_set and generates some labels
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult
from scipy import interpolate
from utilities import load_atomic, convolve
from pyigm.continuum import quasar as pycq
from scipy.special import wofz, erf
import pdb

nHIwav = 15
atmdata = load_atomic(return_HIwav=False)
ww = np.where(atmdata["Ion"] == "1H_I")
HIwav = atmdata["Wavelength"][ww][3:3+nHIwav]
HIfvl = atmdata["fvalue"][ww][3:3+nHIwav]
HIgam = atmdata["gamma"][ww][3:3+nHIwav]


def voigt_tau(par, wavein, logn=True):
    """ Obtain the optical depth for the Lyman series absorption lines
    """
    # Column density
    if logn:
        cold = 10.0 ** par[0]
    else:
        cold = par[0]
    # Redshift
    zp1 = par[1] + 1.0
    tau = np.zeros(wavein.size)
    for ii in range(HIwav.size):
        wv = HIwav[ii]
        # Doppler parameter
        bl = par[2] * wv / 2.99792458E5
        a = HIgam[ii] * wv * wv / (3.76730313461770655E11 * bl)
        cns = wv * wv * HIfvl[ii] / (bl * 2.002134602291006E12)
        cne = cold * cns
        ww = (wavein * 1.0e-8) / zp1
        v = wv * ww * ((1.0 / ww) - (1.0 / wv)) / bl
        tau += cne * wofz(v + 1j * a).real
    return tau


def load_dataset(zem=3.0, snr=0, numspec=20):
    zstr = "zem{0:.2f}".format(zem)
    sstr = "snr{0:d}".format(int(snr))
    extstr = "{0:s}_{1:s}_nspec{2:d}".format(zstr, sstr, numspec)
    fname = "train_data/cnn_qsospec_fluxspec_{0:s}.npy".format(extstr)
    wfdata_all = np.load(fname)
    wdata = wfdata_all[0, :]
    fdata_all = wfdata_all[1:, :]
    zdata_all = np.load("train_data/cnn_qsospec_zvals_{0:s}.npy".format(extstr))
    Ndata_all = np.load("train_data/cnn_qsospec_Nvals_{0:s}.npy".format(extstr))
    bdata_all = np.load("train_data/cnn_qsospec_bvals_{0:s}.npy".format(extstr))
    return fname, fdata_all, wdata, zdata_all, Ndata_all, bdata_all


def generate_continuum(seed, wave, zqso=3.0):
    rstate = np.random.RandomState(seed)
    conti, wfc3_idx = pycq.wfc3_continuum(zqso=zqso, get_orig=True, rstate=rstate)
    convcont = convolve(conti.flux, conti.wavelength, 5000.0)  # Need to smooth out the noisy WFC3 spectra
    cspl = interpolate.interp1d(conti.wavelength, convcont, kind='cubic', bounds_error=False, fill_value="extrapolate")
    return cspl(wave)


def generate_label(ispec, wdata, zdata_all, Ndata_all, bdata_all, zqso=3.0, snr=0, snr_thresh=2.0):
    wlim = (1.0+np.max(zdata_all[ispec, :]))*HIwav[-1]
    labels = np.zeros(wdata.shape[0], dtype=np.int)
    maxodv = np.zeros(wdata.shape[0], dtype=np.int)
    fact = 2.0 * np.sqrt(2.0 * np.log(2.0))
    fc = erf(fact/2.0)  # Fraction of the profile containing the FWHM (i.e. the probability of being within the FWHM)
    dd = ispec
#    for dd in range(nspec):
    print("Preparing labels for spectrum {0:d}/{1:d}".format(dd+1, nspec))
    for zz in range(zdata_all.shape[1]):
        if zdata_all[ispec, zz] == -1:
            # No more lines
            break
        if (1.0+zdata_all[ispec, zz])*HIwav[0] < wlim:
            # Lya line is lower than the highest Lyman series line of the highest redshift absorber
            continue
        par = [Ndata_all[ispec, zz], zdata_all[ispec, zz], bdata_all[ispec, zz]]
        wvpass = HIwav * (1.0 + zdata_all[ispec, zz])
        HIwlm = HIwav[wvpass > wlim]
        HIflm = HIfvl[wvpass > wlim]
        wvpass = wvpass[wvpass > wlim]
        odtmp = voigt_tau(par, wvpass, logn=True)
        for vv in range(wvpass.size):
            amin = np.argmin(np.abs(wdata-wvpass[vv]))
            if odtmp[vv] > maxodv[amin]:
                # Given S/N of spectrum, estimate significance
                EW = 10.0**Ndata_all[ispec, zz] * HIflm[vv] * HIwlm[vv]**2 / 1.13E20
                nsig = snr_thresh  # Record all line positions for perfect data
                if snr > 0:
                    # Estimate the significance of every feature
                    bFWHM = (fact/np.sqrt(2.0)) * bdata_all[ispec, zz]
                    dellam = wvpass[vv] * bFWHM / 299792.458  # Must be in Angstroms
                    nsig = fc * EW * snr / dellam
                if nsig >= snr_thresh:
                    maxodv[amin] = odtmp[vv]
                    labels[amin] = vv+1
    return labels


# Load the data
plotcont = False
plotlabl = False
snr = 0
print("Loading dataset...")
fname, fdata_all, wdata, zdata_all, Ndata_all, bdata_all = load_dataset(3.0, snr)
print("Complete")
nspec = fdata_all.shape[0]
labels = np.zeros((nspec, wdata.shape[0]))
for ispec in range(nspec):
    labels[ispec, :] = generate_label(ispec, wdata, zdata_all, Ndata_all, bdata_all, snr=snr)

# Plot the labels to ensure this has been done correctly
if plotlabl:
    specplot = 1
    cont = generate_continuum(specplot, wdata)
    ymin, ymax = 0.0, np.max(fdata_all[specplot, :])
    plt.plot(wdata, fdata_all[specplot, :], 'k-', drawstyle='steps')
    # Plot Lya
    ww = np.where(labels[specplot, :] == 1)
    plt.vlines(HIwav[0]*(1.0+zdata_all[specplot, :].flatten()), ymin, ymax, 'r', '-')
    plt.plot(wdata[ww], cont[ww], 'ro', drawstyle='steps')
    # Plot Lyb
    ww = np.where(labels[specplot, :] == 2)
    plt.vlines(HIwav[1]*(1.0+zdata_all[specplot, :].flatten()), ymin, ymax, 'g', '--')
    plt.plot(wdata[ww], cont[ww], 'go', drawstyle='steps')
    # Plot Lyg
    ww = np.where(labels[specplot, :] == 3)
    plt.vlines(HIwav[2]*(1.0+zdata_all[specplot, :].flatten()), ymin, ymax, 'b', ':')
    plt.plot(wdata[ww], cont[ww], 'bo', drawstyle='steps')
    plt.show()

print("WARNING :: By continuing, you will save/overwrite the previously stored label data...")
pdb.set_trace()
# Save the labels and the data
print("Saving the labels: Check the following sizes are the same")
print(labels.shape, fdata_all.shape)
np.save(fname.replace(".npy", "_fluxonly.npy"), fdata_all)
np.save(fname.replace(".npy", "_labelonly.npy"), labels)

if plotcont:
    for ispec in range(10):
        cont = generate_continuum(ispec, wdata)
        plt.plot(wdata, fdata_all[ispec, :], 'k-')
        plt.plot(wdata, cont, 'r-')
        plt.show()
        plt.clf()

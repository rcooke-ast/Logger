# This script reads the spectra generated with
# cnn_create_training_set and generates some labels
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult
from scipy import interpolate
from utilities import load_atomic, convolve
from pyigm.continuum import quasar as pycq
from scipy.special import wofz
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
    wfdata_all = np.load("train_data/cnn_qsospec_fluxspec_{0:s}.npy".format(extstr))
    wdata = wfdata_all[0, :]
    fdata_all = wfdata_all[1:, :]
    zdata_all = np.load("train_data/cnn_qsospec_zvals_{0:s}.npy".format(extstr))
    Ndata_all = np.load("train_data/cnn_qsospec_Nvals_{0:s}.npy".format(extstr))
    bdata_all = np.load("train_data/cnn_qsospec_bvals_{0:s}.npy".format(extstr))
    return fdata_all, wdata, zdata_all, Ndata_all, bdata_all


def generate_continuum(seed, wave, zqso=3.0):
    rstate = np.random.RandomState(seed)
    conti, wfc3_idx = pycq.wfc3_continuum(zqso=zqso, get_orig=True, rstate=rstate)
    convcont = convolve(conti.flux, conti.wavelength, 5000.0)  # Need to smooth out the noisy WFC3 spectra
    cspl = interpolate.interp1d(conti.wavelength, convcont, kind='cubic', bounds_error=False, fill_value="extrapolate")
    return cspl(wave)


def generate_label(ispec, fdata_all, wdata, zqso=3.0):
    wlim = (1.0+np.max(zdata_all[ispec, :]))*HIwav[-1]
    cont = generate_continuum(ispec, wdata)
    optdep = -np.log(fdata_all[ispec, :]/cont)
    labels = np.zeros(fdata_all.shape, dtype=np.int)
    for dd in range(nspec):
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
            odtmp = voigt_tau(par, wvpass[wvpass > wlim], logn=True)
            # Given S/N of spectrum
    return labels


# Load the data
fdata_all, wdata, zdata_all, Ndata_all, bdata_all = load_dataset(3.0, 0)
for ispec in range(10):
    cont = generate_continuum(ispec, wdata)
    plt.plot(wdata, fdata_all[ispec, :], 'k-')
    plt.plot(wdata, cont, 'r-')
    plt.show()
    plt.clf()
pdb.set_trace()

nspec = fdata_all.shape[0]

pool = Pool(processes=cpu_count())
async_results = []
for jj in range(nruns):
    seed = np.arange(numspec) + jj * numspec
    async_results.append(pool.apply_async(cnn_numabs, (zem, numseg, numspec, seed, snr)))
pool.close()
pool.join()
map(ApplyResult.wait, async_results)
# Collect the returned data
for jj in range(nruns):
    getVal = async_results[jj].get()
